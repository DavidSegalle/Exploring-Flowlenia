from flowlenia.simutils import Simulator, host_transform_fn_full, host_transform_fn
from flowlenia.flowlenia_params import Config

import jax.random as jr
import numpy as np
from functools import partial
from flowlenia.utils import conn_from_matrix
from flowlenia.environments import init_cross_map, init_connected_cells
from flowlenia.flowlenia_params import beam_mutation

import os

BASE_PATH = os.environ.get("SCRATCH", ".")

do_wall = True
do_connected = False
log_full = False

n_step = 10000
save_interval = 250
X, Y = 512, 512
cluster_padding = 16

if log_full:
    seeds = [0]
    cells = [0]
else:
    seeds = [0]
    cells = [1]

M = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
k = M.sum()
c0, c1 = conn_from_matrix(M)
cfg = Config(X=X, Y=Y, C=3, k=k, c0=c0, c1=c1)
host_transform = host_transform_fn_full if log_full else host_transform_fn

path = BASE_PATH + "/exp_data/walls_debug"


if do_wall:
    for n_cells in cells:
        print(f"Running experiment with {n_cells} walls")
        for seed in seeds:
            mdl, s = init_cross_map(
                cfg,
                seed,
                callback=partial(beam_mutation, sz=20, p=0.01),
                width=8,
                value=-100.0,
                n=n_cells,
                cluster_padding=cluster_padding,
            )
            sim = Simulator(
                mdl,
                f"{path}/{n_cells}_walls/{seed}",
                save_freq=save_interval,
                zip_files=True,
                host_transform_fn=host_transform,
            )
            sim.flush_save_path()
            sim.simulate(s, n_step, jr.key(seed))

if do_connected:
    for n_cells in cells:
        print(f"Running experiment with {n_cells} connected cells")
        for seed in seeds:
            mdl, s = init_connected_cells(
                cfg,
                seed,
                callback=partial(beam_mutation, sz=20, p=0.01),
                width=8,
                value=-100.0,
                n=n_cells,
                passage_width=8,
                cluster_padding=cluster_padding,
            )
            sim = Simulator(
                mdl,
                f"{path}/{n_cells}_connected/{seed}",
                save_freq=save_interval,
                zip_files=True,
                host_transform_fn=host_transform,
            )
            sim.flush_save_path()
            sim.simulate(s, n_step, jr.key(seed))
