from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import chex
from typing import Callable, Optional, Tuple
from jaxtyping import Array, Float
import os

from flowlenia.reintegration_tracking import ReintegrationTracking
from flowlenia.vizutils import display_flp, VideoWriter, img_from_flp
from flowlenia.utils import *

from tqdm import tqdm


@chex.dataclass
class Config:
    """Configuration class for FlowLenia parameters.

    Attributes:
        X (int): The width of the grid.
        Y (int): The height of the grid.
        C (int): The number of channels.
        c0 (jnp.ndarray):
        c1 (jnp.ndarray):
        k (int): The number of kernels.
        dd (int): The diffusion coefficient.
        dt (float): The time step size.
        beta_param (float): The inverse temperature for parameter mixing.
        sigma (float): Upper bound for the flow magnitude.
        n (int):
        theta_A (float):
        border (str): The type of border condition.
        mix_rule (str): The mixing rule.
        wall_mask (Optional[jnp.array]): The wall mask.

    """

    X: int = 128
    Y: int = 128
    C: int = 1
    c0: jnp.ndarray = jnp.array([0])
    c1: jnp.ndarray = jnp.array([[0]])
    k: int = 10
    dd: int = 5
    dt: float = 0.2
    beta_param: float = 1000.0
    sigma: float = 0.65
    n: int = 2
    theta_A: float = 2.0
    border: str = "wall"
    mix_rule: str = "stoch"
    wall_mask: Optional[jnp.array] = None


@chex.dataclass
class State:
    """Represents the state of the FlowLenia model.

    Attributes:
        A (Float[Array, "X Y C"]): Cells activations.
        P (Float[Array, "X Y K"]): Embedded parameters.
        fK (jnp.ndarray): Kernels fft.
        m (jnp.ndarray): Growth parameter.
        s (jnp.ndarray): Growth parameter.
    """

    A: Float[Array, "X Y C"]  # Cells activations
    P: Float[Array, "X Y K"]  # Embedded parameters
    fK: jnp.ndarray  # Kernels fft
    m: jnp.ndarray  # Growth rate
    s: jnp.ndarray  # Growth rate
    Pmix: jnp.ndarray = None  # Mixing parameters


@chex.dataclass
class Params:

    r: jnp.ndarray
    b: jnp.ndarray
    w: jnp.ndarray
    a: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    R: float
    h: jnp.ndarray


class Rule_space:
    def __init__(self, nb_k: int):
        """
        Initialize the Rule_space object.

        Args:
            nb_k (int): number of kernels in the update rule
        """
        self.k = nb_k
        self.kernel_keys = "r b w a m s".split()
        self.spaces = {
            "r": {"low": 0.2, "high": 1.0, "mut_std": 0.2, "shape": None},
            "b": {"low": 0.001, "high": 1.0, "mut_std": 0.2, "shape": (3,)},
            "w": {"low": 0.01, "high": 0.5, "mut_std": 0.2, "shape": (3,)},
            "a": {"low": 0.0, "high": 1.0, "mut_std": 0.2, "shape": (3,)},
            "m": {"low": 0.05, "high": 0.5, "mut_std": 0.2, "shape": None},
            "s": {"low": 0.001, "high": 0.18, "mut_std": 0.01, "shape": None},
            "h": {"low": 0.01, "high": 1.0, "mut_std": 0.2, "shape": None},
            "R": {"low": 2.0, "high": 25.0, "mut_std": 0.2, "shape": None},
        }

    # -----------------------------------------------------------------------------
    def sample(self, key: jnp.ndarray) -> Params:
        """
        Sample a random set of parameters

        Args:
            key (jnp.ndarray): random key
        Returns:
            Params: sampled parameters
        """
        kR, kr, km, ks, ka, kb, kw, kh = jr.split(key, 8)
        R = jr.uniform(kR, (), minval=2.000, maxval=25.0)
        r = jr.uniform(
            kr,
            (self.k,),
            minval=self.spaces["r"]["low"],
            maxval=self.spaces["r"]["high"],
        )
        m = jr.uniform(
            km,
            (self.k,),
            minval=self.spaces["m"]["low"],
            maxval=self.spaces["m"]["high"],
        )
        s = jr.uniform(
            ks,
            (self.k,),
            minval=self.spaces["s"]["low"],
            maxval=self.spaces["s"]["high"],
        )
        a = jr.uniform(
            ka,
            (self.k, 3),
            minval=self.spaces["a"]["low"],
            maxval=self.spaces["a"]["high"],
        )
        b = jr.uniform(
            kb,
            (self.k, 3),
            minval=self.spaces["b"]["low"],
            maxval=self.spaces["b"]["high"],
        )
        w = jr.uniform(
            kw,
            (self.k, 3),
            minval=self.spaces["w"]["low"],
            maxval=self.spaces["w"]["high"],
        )
        h = jr.uniform(
            kh,
            (self.k,),
            minval=self.spaces["h"]["low"],
            maxval=self.spaces["h"]["high"],
        )
        return Params(R=R, r=r, m=m, s=s, a=a, b=b, w=w, h=h)

    def mutate(self, params: Params, key: jnp.ndarray) -> Params:
        """
        Mutate a set of parameters

        Args:
            params (Params): parameters to mutate
            key (jnp.ndarray): random key
        Returns:
            Params: mutated parameters
        """
        kr, km, ks, ka, kb, kw, kh = jr.split(key, 7)
        r = params.r + jr.normal(kr, (self.k,), std=self.spaces["r"]["mut_std"])
        m = params.m + jr.normal(km, (self.k,), std=self.spaces["m"]["mut_std"])
        s = params.s + jr.normal(ks, (self.k,), std=self.spaces["s"]["mut_std"])
        a = params.a + jr.normal(ka, (self.k, 3), std=self.spaces["a"]["mut_std"])
        b = params.b + jr.normal(kb, (self.k, 3), std=self.spaces["b"]["mut_std"])
        w = params.w + jr.normal(kw, (self.k, 3), std=self.spaces["w"]["mut_std"])
        h = params.h + jr.normal(kh, (self.k,), std=self.spaces["h"]["mut_std"])
        # clip
        r = jnp.clip(r, self.spaces["r"]["low"], self.spaces["r"]["high"])
        m = jnp.clip(m, self.spaces["m"]["low"], self.spaces["m"]["high"])
        s = jnp.clip(s, self.spaces["s"]["low"], self.spaces["s"]["high"])
        a = jnp.clip(a, self.spaces["a"]["low"], self.spaces["a"]["high"])
        b = jnp.clip(b, self.spaces["b"]["low"], self.spaces["b"]["high"])
        w = jnp.clip(w, self.spaces["w"]["low"], self.spaces["w"]["high"])
        h = jnp.clip(h, self.spaces["h"]["low"], self.spaces["h"]["high"])
        return Params(R=params.R, r=r, m=m, s=s, a=a, b=b, w=w, h=h)


class FlowLeniaParams:
    def __init__(
        self, cfg: Config, callback: Optional[Callable] = None, *, key: jax.Array
    ):
        """
        Initialize the FlowLeniaParams object.

        Args:
            cfg (Config): The configuration object.
            callback (Optional[Callable]): An optional callback function.
            key (jax.Array): The key used for random number generation.

        Returns:
            None
        """
        self.cfg = cfg
        self.rule_space = Rule_space(cfg.k)
        self.default_params = self.rule_space.sample(key)
        self.RT = ReintegrationTracking(
            cfg,
            has_hidden=True,
        )
        self.clbck = callback

        self.kernel_computer = get_compiled_kernel_computer(cfg.X, cfg.Y, cfg.k)

    # -------------------------------------------------------------------

    @staticmethod
    def default_config():
        return Config

    # -------------------------------------------------------------------

    def __call__(
        self, state: State, key: Optional[jax.Array] = None, no_clbck: bool = False
    ) -> State:
        """
        Perform the main computation of the FlowLenia model.

        Args:
            state (State): The current state of the model.
            key (Optional[jax.Array]): The random key for generating random numbers.
            no_clbck (bool): Flag indicating whether to skip the callback function.

        Returns:
            State: The updated state after the computation.

        """
        kclbck, kflow = jr.split(key, 2)

        A, P = state.A, state.P
        # --- Original Lenia ---
        fA = jnp.fft.fft2(A, axes=(0, 1))  # (x,y,c)

        fAk = fA[:, :, self.cfg.c0]  # (x,y,k)

        U = jnp.real(jnp.fft.ifft2(state.fK * fAk, axes=(0, 1)))  # (x,y,k)

        iU = growth(U, state.m, state.s)

        U = iU * P

        U = jnp.dstack(
            [U[:, :, self.cfg.c1[c]].sum(axis=-1) for c in range(self.cfg.C)]
        )  # (x,y,c)

        # add walls
        if self.cfg.wall_mask is not None:
            U = U + self.cfg.wall_mask[:, :, None]

        # --- FLOW ---

        F = sobel(U)  # (x, y, 2, c) : Flow

        C_grad = sobel(
            A.sum(axis=-1, keepdims=True)
        )  # (x, y, 2, 1) : concentration gradient

        alpha = jnp.clip((A[:, :, None, :] / self.cfg.theta_A) ** self.cfg.n, 0.0, 1.0)

        F = F * (1 - alpha) - C_grad * alpha

        nA, nP, nPmix = self.RT(state, F, kflow)

        # --- Mixing ---
        # Mixing step in the case of the negotiate rule
        # It is separated from the other rules because it reuse the affinity maps (to avoid recomputing them)
        if self.cfg.mix_rule == "negotiate":

            Us = jnp.einsum("xyk, qxyk -> qxyk", iU, nPmix).sum(axis=-1, keepdims=True)

            categorical = jax.random.categorical(
                key, nA.sum(axis=-1, keepdims=True) * Us * self.cfg.beta_param, axis=0
            )
            mask = jax.nn.one_hot(
                categorical, num_classes=(2 * self.cfg.dd + 1) ** 2, axis=-1
            )
            mask = jnp.transpose(mask, (3, 0, 1, 2))
            nP = jnp.sum(nP * mask, axis=0)
            nA = jnp.sum(nA, axis=0)
            nPmix = jnp.sum(nPmix * mask, axis=0)
            state = state.replace(A=nA, P=nP, Pmix=nPmix)
        else:
            state = state.replace(A=nA, P=nP)

        # --- Callback ---

        if self.clbck is not None and not no_clbck:
            state = self.clbck(state, kclbck)

        # ---

        return state

    # -------------------------------------------------------------------

    def rollout(
        self,
        state: State,
        key: Optional[jax.Array] = None,
        steps: int = 100,
    ) -> Tuple[State, State]:
        """
        Perform a rollout of the FlowLenia model while saving intermediate states. Memory heavy.

        Args:
            state (State): The initial state of the model.
            key (Optional[jax.Array]): The random key for generating random numbers.
            steps (int): The number of steps to simulate.

        Returns:
            Tuple[State, State]: The final state and the intermediate states during the simulation.

        """

        def _step(c, x):
            s, k = c
            k, k_ = jr.split(k)
            s = self.__call__(s, k_)
            return [s, k], s

        [s, _], S = jax.lax.scan(_step, [state, key], None, steps)
        return s, S

    # -------------------------------------------------------------------

    def rollout_(
        self, state: State, key: Optional[jax.Array] = None, steps: int = 100
    ) -> State:
        """
        Perform a rollout of the FlowLenia model without saving intermediate states. Lighter on memory.

        Args:
            state (State): The initial state of the model.
            key (Optional[jax.Array]): The random key for generating random numbers.
            steps (int): The number of steps to simulate.

        Returns:
            State: The final state after the simulation.

        """

        def _step(i, c):
            s, k = c
            k, k_ = jr.split(k)
            s = self.__call__(s, k_)
            return (s, k)

        s, _ = jax.lax.fori_loop(0, steps, _step, (state, key))
        return s

    # -------------------------------------------------------------------

    def initialize(self, params: Params = None) -> State:
        """Compute the kernels fft and put dummy arrays as placeholders for A and P"""
        if params is None:
            params = self.default_params
        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C))
        P = jnp.ones((self.cfg.X, self.cfg.Y, self.cfg.k)) * params.h[None, None, :]
        fK = self.kernel_computer(
            params.R,
            params.r,
            params.a,
            params.w,
            params.b,
        )
        Pmix = jnp.ones((self.cfg.X, self.cfg.Y, self.cfg.k)) * params.h[None, None, :]
        return State(A=A, P=P, fK=fK, m=params.m, s=params.s, Pmix=Pmix)

    def rollout_and_save_video(
        self,
        state: State,
        key: Optional[jax.Array] = None,
        steps: int = 100,
        save_path: str = "videos/flowlenia.mp4",
        sub_step_size: int = 100,
        mode: str = "both",
        save_freq: int = 1,
        fps: int = 60,
        downscale_factor: float = 1,
    ):
        def _step(c, x):
            s, k = c
            k, k_ = jr.split(k)
            s = self.__call__(s, k_)
            return [s, k], s

        with VideoWriter(save_path, fps, downscale_factor=downscale_factor) as vid:
            pb = tqdm(range(steps))
            while steps > 0:
                sub_steps = min(sub_step_size, steps)
                [s, k], S = jax.lax.scan(_step, [state, key], None, sub_steps)
                state = s
                key = k
                for i in range(sub_steps):
                    if (steps - i) % save_freq == 0:
                        vid.add(
                            img_from_flp(
                                S.A[i], S.P[i], mode=mode, wall=self.cfg.wall_mask
                            )
                        )
                steps -= sub_steps
                pb.update(sub_steps)
        return state


def beam_mutation(state: State, key: jax.Array, sz: int = 20, p: float = 0.01):
    """Beam mutation callback function.

    Args:
        state (State): The current state of the model.
        key (jax.Array): The random key for generating random numbers.
        sz (int): The size of the beam.
        p (float): The probability of mutation.

    Returns:
        State: The updated state after the mutation.
    """
    kmut, kloc, kp, kmutmix = jr.split(key, 4)
    P = state.P
    Pmix = state.Pmix
    k = P.shape[-1]
    mut = jnp.ones((sz, sz, k)) * jr.normal(kmut, (1, 1, k))
    mutmix = jnp.ones((sz, sz, k)) * jr.normal(kmutmix, (1, 1, k))
    loc = jr.randint(kloc, (3,), minval=0, maxval=P.shape[0] - sz).at[-1].set(0)
    dP = jax.lax.dynamic_update_slice(jnp.zeros_like(P), mut, loc)
    dPmix = jax.lax.dynamic_update_slice(jnp.zeros_like(Pmix), mutmix, loc)
    m = (jr.uniform(kp, ()) < p).astype(float)
    P = P + dP * m
    Pmix = Pmix + dPmix * m
    return state.replace(P=P, Pmix=Pmix)


if __name__ == "__main__":
    from time import time

    save_freq = 1
    substep_size = 20
    mix_rule = "energy"
    n_seeds = 1
    init_size = 40

    t = time()
    for seed in range(n_seeds):
        save_path = f"videos/flowlenia_temp/{seed}.mp4"
        n_step = 500
        M = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
        k = M.sum()
        cfg = Config(X=128, Y=128, C=3, k=k, mix_rule=mix_rule, border="torus")
        c0, c1 = conn_from_matrix(M)
        cfg = cfg.replace(c0=c0, c1=c1)
        flp = FlowLeniaParams(cfg, key=jr.key(seed), callback=None)
        s = flp.initialize()
        locs = jnp.arange(init_size) + (cfg.X // 2 - init_size // 2)
        A = s.A.at[jnp.ix_(locs, locs)].set(
            jr.uniform(jr.key(seed), (init_size, init_size, 3))
        )
        s = s.replace(A=A)
        s = flp.rollout_and_save_video(
            s,
            jr.key(seed),
            steps=n_step,
            save_path=save_path,
            mode="state",
            save_freq=save_freq,
            sub_step_size=substep_size,
        )
    os.system(f"ffplay -autoexit -fs {save_path}")
    print(f"Elapsed time: {time()-t:.2f}s")
