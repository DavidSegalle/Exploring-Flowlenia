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

n_step = 100 * 100
save_interval = 25
X, Y = 512, 512
cluster_padding = 8


M = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
k = M.sum()
c0, c1 = conn_from_matrix(M)
host_transform = host_transform_fn_full
n_cells = 3

path = BASE_PATH + "/exp_data/distribution"


# mixing_rules = ["stoch_softmax", "stoch_softmax_gene_wise"]
mixing_rules = ["negotiate"]
beta_params = [100.0]


def init_corner_activation(s, size=64):
    mask = np.zeros_like(s.A)
    mask[:size, :size] = 1
    new_A = s.A * mask
    return s.replace(A=new_A)


for mixing_rule in mixing_rules:
    for beta_param in beta_params:
        cfg = Config(
            X=X,
            Y=Y,
            C=3,
            k=k,
            c0=c0,
            c1=c1,
            beta_param=beta_param,
            mix_rule=mixing_rule,
        )
        print(
            f"Running experiment with {mixing_rule} mixing rules and beta_param {beta_param}"
        )
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
            s = init_corner_activation(s)
            sim = Simulator(
                mdl,
                f"{path}/{mixing_rule}_{int(beta_param)}_{n_cells}cells/{seed}",
                save_freq=save_interval,
                zip_files=True,
                host_transform_fn=host_transform,
                video_pth=f"{path}/{mixing_rule}_{int(beta_param)}_{n_cells}cells/videos/video_{seed}.mp4",
            )
            sim.flush_save_path()
            sim.simulate(s, n_step, jr.key(seed))

print("Time:", time() - t)
