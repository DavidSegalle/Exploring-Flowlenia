from flowlenia.analysis.ea import compute_nnea_online

import numpy as np
from flowlenia.utils import conn_from_matrix

import pandas as pd

import os


save_interval = 250

M = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]])
k = M.sum()
c0, c1 = conn_from_matrix(M)
n_cells = 0


BASE_PATH = os.environ.get("SCRATCH", ".")
path = BASE_PATH + "/exp_data/mixing_negoparams"

mixing_rules = ["negotiate"]
# mixing_rules = ["avg", "softmax"]
beta_params = [100]
subsampling_rate = int(os.environ.get("SLURM_ARRAY_TASK_ID", "1"))

seeds = [0, 1, 2, 3, 4]
for mixing_rule in mixing_rules:
    for beta_param in beta_params:
        ea_data = pd.DataFrame(
            columns=["seed", "Mixing rule", "Beta", "t", "A_n", "delta_n"]
        )
        for seed in seeds:
            if mixing_rule in ["stoch_softmax", "stoch_softmax_gene_wise", "negotiate"]:
                print(
                    f"Running seed {seed} for {mixing_rule} mixing rule and beta_param {beta_param}"
                )
                A_n, delta_n, delta_p = compute_nnea_online(
                    f"{path}/{mixing_rule}_{int(beta_param)}_{n_cells}cells/{seed}",
                    subsampling_rate=subsampling_rate,
                )
            else:
                print(f"Running seed {seed} for {mixing_rule} mixing rule")
                A_n, delta_n, delta_p = compute_nnea_online(
                    f"{path}/{mixing_rule}/{seed}",
                    subsampling_rate=subsampling_rate,
                )
            for t in range(len(A_n)):
                row = {
                    "seed": seed,
                    "Mixing rule": mixing_rule,
                    "Beta": beta_param,
                    "t": t * save_interval * subsampling_rate,
                    "A_n": A_n[t],
                    "delta_n": delta_n[t],
                    "delta_p": delta_p[t],
                }
                ea_data = pd.concat([ea_data, pd.DataFrame([row])], ignore_index=True)

        if mixing_rule in ["stoch_softmax", "stoch_softmax_gene_wise", "negotiate"]:
            ea_data.to_csv(
                f"{path}/ea_data_{mixing_rule}_{int(beta_param)}{n_cells}cells_srate{subsampling_rate}.csv",
                index=False,
            )
        else:
            ea_data.to_csv(
                f"{path}/ea_data_{mixing_rule}_srate{subsampling_rate}.csv",
                index=False,
            )
            break
