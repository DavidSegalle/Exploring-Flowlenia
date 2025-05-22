from time import time

t = time()

from flowlenia.simutils import (
    Simulator,
    host_transform_fn_full,
    host_transform_fn,
    host_transform_fn_light,
)
from flowlenia.flowlenia_params import Config

import jax
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

n_step = 100000
save_interval = 250
X, Y = 2048, 2048
cluster_padding = 96
downscale_factor = 1


M = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
k = M.sum()
c0, c1 = conn_from_matrix(M)
host_transform = host_transform_fn_light
n_cells = 0

path = BASE_PATH + f"/exp_data/large_map/{k}_kernels/{X}x{Y}"

# mixing_rules = ["avg", "softmax", "stoch", "stoch_gene_wise","dot"]
mixing_rules = ["stoch_gene_wise"]

for mixing_rule in mixing_rules:
    cfg = Config(X=X, Y=Y, C=3, k=k, c0=c0, c1=c1, mix_rule=mixing_rule)
    print(f"Running experiment with {mixing_rule} mixing rule")
    for seed in seeds:
        mdl, s = init_connected_cells(
            cfg,
            seed,
            callback=partial(beam_mutation, sz=10, p=0.05),
            width=16,
            value=-100.0,
            n=n_cells,
            passage_width=16,
            cluster_padding=cluster_padding,
        )
        sim = Simulator(
            mdl,
            f"{path}/{mixing_rule}/beta{int(cfg.beta_param)}/{seed}",
            save_freq=save_interval,
            zip_files=True,
            host_transform_fn=host_transform,
            video_pth=f"{path}/{mixing_rule}/beta{int(cfg.beta_param)}/videos/video_{seed}.mp4",
            video_downscale_factor=downscale_factor,
            video_low_ram_mode=True,
        )
        sim.flush_save_path()
        sim.simulate(s, n_step, jr.key(seed))


print("Time:", time() - t)
