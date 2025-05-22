import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import os
import pickle
from flowlenia.optimization.evoflow import EvoFlow
from flowlenia.flowlenia_params import Config as FL_Config, State as FL_State
from flowlenia.optimization.tasks import EvoFlow_DirectedMotion
from flowlenia.optimization.optutils import center_of_mass
from flowlenia.optimization.evoflow import EvoParams, EvoFlow
from flowlenia.optimization.optexp import make_expe, ExperimentConfig
from functools import partial
import shutil
from flowlenia.utils import conn_from_matrix
import numpy as np


class EvoFlow_Metamorphose(EvoFlow):

    def _build_evaluator(self):
        ker_comp_map = jax.vmap(self.fl.kernel_computer)
        com = partial(center_of_mass, X=self.fl_config.X, Y=self.fl_config.Y)
        com = jax.vmap(com)
        X, Y, C = self.fl_config.X, self.fl_config.Y, self.fl_config.C
        mx, my = X // 2, Y // 2
        popsize = self.es_config["popsize"]

        rollout_fn = jax.vmap(lambda s, k: self.fl.rollout(s, k, 400)[0])

        pattern_end = -jnp.ones((X, Y, C))
        pattern_end = pattern_end.at[mx - 40 : mx + 40, my - 2 : my + 2, :].set(1)
        pattern_end = pattern_end.at[mx - 2 : mx + 2, my - 40 : my + 40, :].set(1)
        pattern_init = -jnp.ones((X, Y, C))
        pattern_init = pattern_init.at[mx - 20 : mx + 20, my - 20 : my + 20, :].set(1)
        pattern_init = pattern_init.at[mx - 10 : mx + 10, my - 10 : my + 10, :].set(-1)

        @jax.jit
        def evaluate(params: EvoParams, key: jnp.ndarray):
            fKs = ker_comp_map(
                params.R,
                params.r,
                params.a,
                params.w,
                params.b,
            )
            As = (
                jnp.zeros((popsize, X, Y, C))
                .at[:, mx - 20 : mx + 20, my - 20 : my + 20, :]
                .set(params.init)
            )
            Ps = (
                jnp.ones((popsize, X, Y, self.fl_config.k)) * params.h[:, None, None, :]
            )
            init_states = FL_State(A=As, P=Ps, fK=fKs, m=params.m, s=params.s)

            finals = rollout_fn(init_states, jr.split(key, popsize))
            # coms = com(jnp.sum(finals.A, axis=-1, keepdims=True))
            cost = jnp.sum(finals.A * pattern_end, axis=(1, 2, 3)) + jnp.sum(
                init_states.A * pattern_init, axis=(1, 2, 3)
            )
            return cost

        return evaluate


if __name__ == "__main__":
    BASE_PATH = os.environ.get("SCRATCH", ".")
    SAVE_DIR = f"{BASE_PATH}/exp_data/metamorph/data"  # @param {type:"string"}
    os.makedirs(SAVE_DIR, exist_ok=True)

    SAVE_RATE = 100  # @param {type:"integer"}
    population_size = 16  # @param {type:"integer"}
    optimizer = "adam"  # @param {type:"string"}
    learning_rate = 0.01  # @param
    seed = 42  # @param {type:"integer"}
    generations = 5001  # @param {type:"integer"}
    n_seeds = 1  # @param {type:"integer"}

    SEED = jax.random.PRNGKey(seed)

    ES_CONFIG = {
        "popsize": population_size,
        "opt_name": optimizer,
        "lrate_init": learning_rate,
    }

    C = 3
    M = np.array([[3, 1, 1], [1, 3, 1], [1, 1, 3]])
    k = M.sum()
    cfg = FL_Config(X=128, Y=128, C=3, k=k, mix_rule="negotiate")
    c0, c1 = conn_from_matrix(M)
    cfg = cfg.replace(c0=c0, c1=c1)

    config = ExperimentConfig(
        name="metamorph2",
        task=EvoFlow_Metamorphose,
        fl_config=cfg,
        es_config=ES_CONFIG,
        seed=SEED,
        n_seeds=n_seeds,
        generations=generations,
    )

    make_expe(config, SAVE_DIR, save_rate=SAVE_RATE)
