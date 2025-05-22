from functools import partial
from flowlenia.flowlenia_params import (
    FlowLeniaParams,
    Config as ConfigParams,
    State as StateParams,
)
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Float
import numpy as np
import chex

from flowlenia.vizutils import display_flpf
from flowlenia.utils import conn_from_matrix_food


@chex.dataclass
class State(StateParams):
    FM: Float[Array, "X Y T"]  # Food map


@chex.dataclass
class Config(ConfigParams):
    n_food: int = 1
    rho_decay: Float[Array, "C"] = jnp.array([0.99, 0.99, 0.99])
    rho_digest: Float[Array, "T C"] = jnp.array([[0.1, 0, 0]])


class FlowLeniaFood(FlowLeniaParams):
    def __call__(
        self, state: State, key: jax.Array | None = None, no_clbck: bool = False
    ):
        A, FM = state.A, state.FM
        # Food / matter exchange
        nA = A * self.cfg.rho_decay
        m_growth = jnp.clip(jnp.dot(nA, self.cfg.rho_digest.transpose()), 0, FM)
        nA = nA.at[:, :, 0].add(m_growth.sum(-1))
        nFM = FM.at[:, :, :].add(-m_growth)
        nA = jax.lax.concatenate([nA, nFM], dimension=2)

        state = super().__call__(state.replace(A=nA, FM=nFM), key, no_clbck=True)

        nA = state.A[:, :, : -self.cfg.n_food]
        nFM = state.A[:, :, -self.cfg.n_food :]
        state = state.replace(A=nA, FM=nFM)

        if not no_clbck and self.clbck is not None:
            state = self.clbck(state, key)

        return state

    def initialize(self, params=None) -> State:
        if params is None:
            params = self.default_params
        s = super().initialize(params)
        A = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.C - self.cfg.n_food))
        FM = jnp.zeros((self.cfg.X, self.cfg.Y, self.cfg.n_food))
        return State(A=A, P=s.P, fK=s.fK, FM=FM, m=s.m, s=s.s)


def beam_food(state: State, key: jax.Array, sz: int = 20, p: float = 0.01):
    kconc, kloc, kft, kp = jr.split(key, 4)
    FM = state.FM
    mut = jnp.ones((sz, sz)) * jr.uniform(kconc, (1, 1))
    loc = jr.randint(kloc, (2,), minval=0, maxval=FM.shape[0] - sz)
    ft = jr.randint(kft, (1,), minval=0, maxval=FM.shape[-1])
    dFM = jax.lax.dynamic_update_slice(jnp.zeros_like(FM[:, :, 0]), mut, loc)[
        :, :, None
    ]
    m = (jr.uniform(kp, ()) < p).astype(float)
    FM = FM.at[..., ft].add(dFM * m)
    return state.replace(FM=FM)


if __name__ == "__main__":
    seed = 11
    n_step = 500

    M = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
    food_viz = np.array([[1, 0, 0]])
    k = M.sum() + food_viz.sum()
    c0, c1 = conn_from_matrix_food(M, food_viz)
    cfg = Config(
        X=128,
        Y=128,
        C=3 + food_viz.shape[0],
        k=k,
        rho_digest=jnp.array([[0.1, 0, 0]]),
        n_food=1,
    )
    cfg = cfg.replace(c0=c0, c1=c1)
    flf = FlowLeniaFood(
        cfg, key=jr.key(seed), callback=partial(beam_food, sz=20, p=0.05)
    )
    s = flf.initialize()
    locs = jnp.arange(20) + (cfg.X // 2 - 10)
    food_locs_x = jnp.arange(50) + (cfg.X // 2 - 50)
    food_locs_y = jnp.arange(20) + (cfg.Y // 2 - 10)
    A = s.A.at[jnp.ix_(locs, locs)].set(jr.uniform(jr.key(2), (20, 20, 3)))
    P = s.P.at[jnp.ix_(locs, locs)].set(
        jnp.ones((20, 20, k)) * jr.uniform(jr.key(111), (1, 1, k))
    )
    FM = s.FM.at[jnp.ix_(food_locs_x, food_locs_y)].set(jnp.ones((50, 20, 1)))
    s = s.replace(A=A, P=P, FM=FM)
    s, S = flf.rollout(s, jr.key(1), n_step)
    display_flpf(S, n_step)
