import jax
import jax.experimental
import jax.numpy as jnp
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import partial


@jax.jit
def add_trees(t1, t2):
    return jax.tree_map(lambda x1, x2: x1 + x2, t1, t2)


@jax.jit
def clip_tree(tree, lower, upper):
    return jax.tree_map(jnp.clip, tree, lower, upper)


@partial(jax.jit, static_argnums=(1,))
def unstack_tree(tree, dims):
    return [jax.tree_map(lambda x: x[i], tree) for i in range(dims)]


def center_of_mass(im, X, Y):
    """
    im: array (W, H, C)
    X: int (width of image)
    Y: int (height of image)
    """
    im = im[:, :, 0]
    mass = im.sum()
    x, y = jnp.arange(X), jnp.arange(Y)
    xx, yy = jnp.meshgrid(x, y)
    iX, iY = xx - X / 2, yy - Y / 2

    # Centroids
    cx = (iX * im).sum() / (mass + 1e-10)
    cy = (iY * im).sum() / (mass + 1e-10)

    z = jnp.zeros(2)
    z = z.at[0].set(cx / X)
    z = z.at[1].set(cy / Y)

    return z


def ring_sampling(key, dmin, dmax, n=30):
    ang_key, dist_key = jax.random.split(key)
    angles = jax.random.uniform(
        ang_key, shape=(n,), minval=0, maxval=2 * np.pi
    )  # in radians
    dists = jax.random.uniform(dist_key, shape=(n,), minval=dmin, maxval=dmax)
    return jnp.dstack([dists * jnp.cos(angles), dists * jnp.sin(angles)])[0]


def save_scan(chkpt_file, save_rate=20, aux_infos={}):
    """Decorator for scanned func
    Saves carried state of scanned function every save_rate steps"""

    def _save(it_data):
        iter, data = it_data
        with open(f"{chkpt_file}_{iter}.pickle", "wb") as handle:
            pickle.dump({**data, **aux_infos}, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save(iter, carry, y):
        _ = jax.lax.cond(
            iter % save_rate == 0,
            lambda _: jax.experimental.io_callback(
                _save, (), [iter, {"carry": carry, "y": y}]
            ),
            lambda _: (),
            operand=None,
        )

    def _save_scan(func):

        def _wrapped_func(carry, x):
            if type(x) is tuple:
                iter, *_ = x
            else:
                iter = x
            n_carry, y = func(carry, x)
            save(iter, carry, y)
            return n_carry, y

        return _wrapped_func

    return _save_scan


def plot_scan(save_file, plot_rate=20, xlabel="", ylabel="", title="", **scatter_kws):
    xs = []
    ys = []

    def _plot(xy, *_):
        x, y = xy
        xs.append(x)
        ys.append(y)
        if not x % plot_rate:
            plt.figure()
            plt.scatter(xs, ys, **scatter_kws)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(title)
            plt.savefig(save_file)

    def plot(iter, fits):
        _ = jax.experimental.io_callback(_plot, (), [iter, fits])
        return _

    def _plot_scan(func):

        def _wrapped(carry, x):
            if type(x) is tuple:
                iter, *_ = x
            else:
                iter = x
            carry, ys = func(carry, x)
            plot(x, jnp.max(ys))
            # host_callback.barrier_wait()
            return carry, ys

        return _wrapped

    return _plot_scan


def progress_bar_scan(num_samples):
    "Progress bar for a JAX scan"
    bar = tqdm(total=num_samples)

    def _update_tqdm(arg):
        bar.update(1)

        _ = jax.experimental.io_callback(_update_tqdm, (), None)

    def _close_tqdm(arg, transform):
        bar.close()

    def _progress_bar_scan(func):
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
        Note that `body_fun` must either be looping over `np.arange(num_samples)`,
        or be looping over a tuple who's first element is `np.arange(num_samples)`
        This means that `iter_num` is the current iteration number
        """

        def wrapper_progress_bar(carry, x):
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x
            _ = jax.experimental.io_callback(_update_tqdm, (), 1)
            carry, y = func(carry, x)
            _ = jax.lax.cond(
                iter_num == num_samples - 1,
                lambda _: jax.experimental.io_callback(_close_tqdm, (), None),
                lambda _: (),
                operand=None,
            )
            return carry, y

        return wrapper_progress_bar

    return _progress_bar_scan
