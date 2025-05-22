from flowlenia.flowlenia_params import FlowLeniaParams, Config, State, beam_mutation
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flowlenia.utils import conn_from_matrix
from flowlenia.vizutils import display_flp


seed = 111
n_step = 500
M = np.array([[3, 1, 0], [0, 3, 1], [1, 0, 3]])
k = M.sum()
cfg = Config(X=128, Y=128, C=3, k=k)
c0, c1 = conn_from_matrix(M)
cfg = cfg.replace(c0=c0, c1=c1)
flp = FlowLeniaParams(
    cfg, key=jr.key(seed), callback=partial(beam_mutation, sz=20, p=0.1)
)
s = flp.initialize()
locs = jnp.arange(20) + (cfg.X // 2 - 10)
A = s.A.at[jnp.ix_(locs, locs)].set(jr.uniform(jr.key(seed), (20, 20, 3)))
P = s.P.at[jnp.ix_(locs, locs)].set(
    jnp.ones((20, 20, k)) * jr.uniform(jr.key(seed), (1, 1, k))
)
s = s.replace(A=A, P=P)
s, S = flp.rollout(s, jr.key(seed), n_step)
display_flp(S, n_step=n_step, mode="state")
