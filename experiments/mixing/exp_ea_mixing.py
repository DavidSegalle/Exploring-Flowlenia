from time import time

t = time()

from flowlenia.simutils import (
    Simulator,
    host_transform_fn_full,
    host_transform_fn,
    host_transform_fn_light,
)
from flowlenia.flowlenia_params import Config

import jax.random as jr
import numpy as np
from functools import partial
from flowlenia.utils import conn_from_matrix
from flowlenia.environments import init_cross_map, init_connected_cells
from flowlenia.flowlenia_params import beam_mutation

import os


BASE_PATH = os.environ.get("SCRATCH", ".")

# seed = 0
seed = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
seeds = [seed]

n_step = 100 * 1000
save_interval = 250
X, Y = 512, 512
cluster_padding = 16


M = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
k = M.sum()
c0, c1 = conn_from_matrix(M)
host_transform = host_transform_fn_full
n_cells = 3

path = BASE_PATH + "/exp_data/mixing_additional"

# mixing_rules = ["avg", "softmax", "stoch", "stoch_gene_wise","dot"]
mixing_rules = ["stoch_gene_wise"]

for mixing_rule in mixing_rules:
    cfg = Config(X=X, Y=Y, C=3, k=k, c0=c0, c1=c1, mix_rule=mixing_rule)
    print(f"Running experiment with {mixing_rule} mixing rules")
    for seed in seeds:
        mdl, s = init_connected_cells(
                cfg,
                seed,
                callback=partial(beam_mutation, sz=10, p=0.01),
                width=8,
                value=-100.0,
                n=n_cells,
                passage_width=8,
                cluster_padding=cluster_padding,
            )
        sim = Simulator(
                mdl,
                f"{path}/{mixing_rule}_{int(beta_param)}/{seed}",
                save_freq=save_interval,
                zip_files=True,
                host_transform_fn=host_transform,
                video_pth=f"{path}/{mixing_rule}_{int(beta_param)}/videos/video_{seed}.mp4",
            )
        sim.flush_save_path()
        sim.simulate(s, n_step, jr.key(seed))

print("Time:", time() - t)
