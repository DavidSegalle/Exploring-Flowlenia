import numpy as np
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.debug import callback
from tqdm.auto import tqdm
import typing


def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)


ker_f = lambda x, a, w, b: (b * jnp.exp(-((x[..., None] - a) ** 2) / w)).sum(-1)

bell = lambda x, m, s: jnp.exp(-(((x - m) / s) ** 2) / 2)


def growth(U, m, s):
    return bell(U, m, s) * 2 - 1


kx = jnp.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])
ky = jnp.transpose(kx)


def sobel_x(A):
    """
    A : (x, y, c)
    ret : (x, y, c)
    """
    return jnp.dstack(
        [jsp.signal.convolve2d(A[:, :, c], kx, mode="same") for c in range(A.shape[-1])]
    )


def sobel_y(A):
    return jnp.dstack(
        [jsp.signal.convolve2d(A[:, :, c], ky, mode="same") for c in range(A.shape[-1])]
    )


@jax.jit
def sobel(A):
    return jnp.concatenate(
        (sobel_y(A)[:, :, None, :], sobel_x(A)[:, :, None, :]), axis=2
    )


def get_kernels_fft(X, Y, k, R, r, a, w, b):
    mid = X // 2
    Ds = [
        jnp.linalg.norm(jnp.mgrid[-mid:mid, -mid:mid], axis=0) / ((R + 15) * r[k])
        for k in range(k)
    ]  # (x,y,k)
    K = jnp.dstack(
        [
            sigmoid(-(D - 1) * 10) * ker_f(D, a[k], w[k], b[k])
            for k, D in zip(range(k), Ds)
        ]
    )
    nK = K / jnp.sum(K, axis=(0, 1), keepdims=True)  # Normalize kernels
    fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0, 1)), axes=(0, 1))  # Get kernels fft

    return fK


def get_compiled_kernel_computer(X: int, Y: int, nb_k: int):
    def compute_kernels(R, r, a, w, b):
        return get_kernels_fft(X, Y, nb_k, R, r, a, w, b)

    return jax.jit(compute_kernels)


def get_kernels(X: int, Y: int, nb_k: int, params):
    mid = X // 2
    Ds = [
        np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0)
        / ((params["R"] + 15) * params["r"][k])
        for k in range(nb_k)
    ]  # (x,y,k)
    K = jnp.dstack(
        [
            sigmoid(-(D - 1) * 10)
            * ker_f(D, params["a"][k], params["w"][k], params["b"][k])
            for k, D in zip(range(nb_k), Ds)
        ]
    )
    nK = K / jnp.sum(K, axis=(0, 1), keepdims=True)
    return nK


def conn_from_matrix(mat):
    C = mat.shape[0]
    c0 = []
    c1 = [[] for _ in range(C)]
    i = 0
    for s in range(C):
        for t in range(C):
            n = mat[s, t]
            if n:
                c0 = c0 + [s] * n
                c1[t] = c1[t] + list(range(i, i + n))
            i += n
    return c0, c1


def conn_from_matrix_food(mat, food_viz):
    n_food = len(food_viz)
    k = mat.shape[0]
    nmat = np.zeros((k + n_food, k + n_food), dtype=int)
    nmat[:k, :k] = mat
    nmat[k:, :k] = food_viz
    return conn_from_matrix(nmat)


def conn_from_lists(c0, c1, C):
    return c0, [[i == c1[i] for i in range(len(c0))] for _ in range(C)]


def scan_tqdm(
    n: int,
    print_rate: typing.Optional[int] = None,
    **kwargs,
) -> typing.Callable:
    """
    tqdm progress bar for a JAX scan

    Parameters
    ----------
    n : int
        Number of scan steps/iterations.
    print_rate : int
        Optional integer rate at which the progress bar will be updated,
        by default the print rate will 1/20th of the total number of steps.
    **kwargs
        Extra keyword arguments to pass to tqdm.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """

    _update_progress_bar, close_tqdm = build_tqdm(n, print_rate, **kwargs)

    def _scan_tqdm(func):
        """Decorator that adds a tqdm progress bar to `body_fun` used in `jax.lax.scan`.
        Note that `body_fun` must either be looping over `jnp.arange(n)`,
        or be looping over a tuple who's first element is `jnp.arange(n)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _update_progress_bar(iter_num)
            result = func(carry, x)
            return close_tqdm(result, iter_num)

        return wrapper_progress_bar

    return _scan_tqdm


# From https://github.com/jeremiecoullon/jax-tqdm
def loop_tqdm(
    n: int,
    print_rate: typing.Optional[int] = None,
    **kwargs,
) -> typing.Callable:
    """
    tqdm progress bar for a JAX fori_loop

    Parameters
    ----------
    n : int
        Number of iterations.
    print_rate: int
        Optional integer rate at which the progress bar will be updated,
        by default the print rate will 1/20th of the total number of steps.
    **kwargs
        Extra keyword arguments to pass to tqdm.

    Returns
    -------
    typing.Callable:
        Progress bar wrapping function.
    """

    _update_progress_bar, close_tqdm = build_tqdm(n, print_rate, **kwargs)

    def _loop_tqdm(func):
        """
        Decorator that adds a tqdm progress bar to `body_fun`
        used in `jax.lax.fori_loop`.
        """

        def wrapper_progress_bar(i, val):
            _update_progress_bar(i)
            result = func(i, val)
            return close_tqdm(result, i)

        return wrapper_progress_bar

    return _loop_tqdm


def build_tqdm(
    n: int,
    print_rate: typing.Optional[int],
    **kwargs,
) -> typing.Tuple[typing.Callable, typing.Callable]:
    """
    Build the tqdm progress bar on the host
    """

    desc = kwargs.pop("desc", f"Running for {n:,} iterations")
    message = kwargs.pop("message", desc)
    for kwarg in ("total", "mininterval", "maxinterval", "miniters"):
        kwargs.pop(kwarg, None)

    tqdm_bars = {}

    if print_rate is None:
        if n > 20:
            print_rate = min(int(n / 20), 250)
        else:
            print_rate = 1
    else:
        if print_rate < 1:
            raise ValueError(f"Print rate should be > 0 got {print_rate}")
        elif print_rate > n:
            raise ValueError(
                "Print rate should be less than the "
                f"number of steps {n}, got {print_rate}"
            )

    remainder = n % print_rate

    def _define_tqdm(arg, transform):
        tqdm_bars[0] = tqdm(range(n), **kwargs)
        tqdm_bars[0].set_description(message, refresh=False)

    def _update_tqdm(arg, transform):
        tqdm_bars[0].update(int(arg))

    def _update_progress_bar(iter_num):
        "Updates tqdm from a JAX scan or loop"
        _ = jax.lax.cond(
            iter_num == 0,
            lambda _: callback(_define_tqdm, None, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm every multiple of `print_rate` except at the end
            (iter_num % print_rate == 0) & (iter_num != n - remainder),
            lambda _: callback(_update_tqdm, print_rate, None, ordered=True),
            lambda _: None,
            operand=None,
        )

        _ = jax.lax.cond(
            # update tqdm by `remainder`
            iter_num == n - remainder,
            lambda _: callback(_update_tqdm, remainder, None, ordered=True),
            lambda _: None,
            operand=None,
        )

    def _close_tqdm(arg, transform):
        tqdm_bars[0].close()

    def close_tqdm(result, iter_num):
        _ = jax.lax.cond(
            iter_num == n - 1,
            lambda _: callback(_close_tqdm, None, None, ordered=True),
            lambda _: None,
            operand=None,
        )
        return result

    return _update_progress_bar, close_tqdm
