from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jr
import os

from flowlenia.vizutils import display_flp
from flowlenia.utils import *
from flowlenia.environments import init_cross_map, init_connected_cells
from flowlenia.flowlenia_params import FlowLeniaParams, Config, beam_mutation

seed = 0
n_step = 1000
X, Y = 512, 512
incremental_save = True
save_freq = 1
substep_size = 20
mix_rule = "energy"
save_path = f"videos/flowlenia_wall_{mix_rule}_{save_freq}_temp.mp4"

M = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
k = M.sum()
c0, c1 = conn_from_matrix(M)
cfg = Config(X=X, Y=Y, C=3, k=k, c0=c0, c1=c1, mix_rule=mix_rule, beta_param=100.0)
flp, s = init_connected_cells(
    cfg,
    seed,
    callback=partial(beam_mutation, sz=20, p=0.1),
    width=8,
    value=-100.0,
    n=3,
    passage_width=8,
    cluster_padding=16,
    n_clusters=4,
)
if incremental_save:
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
else:
    s, S = flp.rollout(s, jr.key(seed), n_step)
    display_flp(
        S, n_step=n_step, mode="both", wall=flp.cfg.wall_mask, save_path=save_path
    )
