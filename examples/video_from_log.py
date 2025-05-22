from flowlenia.simutils import load_and_make_video
from flowlenia.flowlenia_params import Config

import numpy as np
from functools import partial
from flowlenia.utils import conn_from_matrix
from flowlenia.environments import init_cross_map, init_connected_cells
from flowlenia.flowlenia_params import beam_mutation

import os

seed = 0
n_step = 1000
save_interval = 100
X, Y = 512, 512
incremental_save = True
mode = "both"
save_path = f"flowlenia_ea_{mode}_temp.mp4"
data_path = "/gpfsscratch/rech/imi/uua17jd/exp_data/mixing/full/stoch_gene_wise/0"
# data_path = "exp_data/walls_debug/1_connected/1"
show_video = True
plot_data = False

M = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
k = M.sum()
c0, c1 = conn_from_matrix(M)
cfg = Config(X=X, Y=Y, C=3, k=k, c0=c0, c1=c1)
mdl, s = init_connected_cells(
    cfg,
    seed,
    callback=partial(beam_mutation, sz=20, p=0.001),
    width=8,
    value=-100.0,
    n=0,
    passage_width=8,
)
if show_video:
    load_and_make_video(data_path, save_path, mode=mode, wall=mdl.cfg.wall_mask)
    # show video in video player
    # os.system(f"ffplay -autoexit -fs {save_path}")
