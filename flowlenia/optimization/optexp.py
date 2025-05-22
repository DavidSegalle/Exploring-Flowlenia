import chex
import jax
import jax.numpy as jnp
import os
import pickle
from flowlenia.optimization.evoflow import EvoFlow
from flowlenia.flowlenia_params import Config as FL_Config
from flowlenia.optimization.tasks import EvoFlow_DirectedMotion


@chex.dataclass
class ExperimentConfig:
    name: str
    task: EvoFlow
    fl_config: FL_Config
    es_config: dict
    seed: jnp.ndarray
    n_seeds: int
    generations: int


def make_expe(config: ExperimentConfig, save_dir, save_rate=100):

    exp_name = config.name
    chkpt_file = f"{save_dir}/{exp_name}/{exp_name}"

    # Create directory for saves--------------------------------------
    if not os.path.isdir(f"{save_dir}/{exp_name}"):
        os.mkdir(f"{save_dir}/{exp_name}")
        start_seed = 0
        chkpt = None
        # Save config-----------------------------------------------------
        with open(f"{chkpt_file}_config.pickle", "wb") as handle:
            pickle.dump(config, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        # Get last chkpt
        start_seed = 0
        for seed in range(config.n_seeds):
            if os.path.isfile(f"{chkpt_file}_{seed}_results.pickle"):
                start_seed += 1
            else:
                break
        chkpt = None

    # Initialize es---------------------------------------------------
    es = config.task(config.fl_config, config.es_config)

    # Launch runs-----------------------------------------------------
    seeds = jax.random.split(config.seed, config.n_seeds)
    for i in range(start_seed, config.n_seeds):
        key = seeds[i]
        data = es.run_fn(
            key,
            config.generations,
            chkpt_file=f"{chkpt_file}_{i}",
            save_rate=save_rate,
            name=f"{exp_name}_{i}",
            chkpt=chkpt,
        )
        with open(f"{chkpt_file}_{i}_results.pickle", "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save final results-----------------------------------------------
    for s in range(config.n_seeds):
        tmp = []
        with open(f"{chkpt_file}_{s}_results.pickle", "rb") as f:
            tmp.append(pickle.load(f))
    with open(f"{chkpt_file}_results.pickle", "wb") as handle:
        pickle.dump(jnp.stack(tmp), handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    SAVE_DIR = "experiments"  # @param {type:"string"}
    SAVE_RATE = 100  # @param {type:"integer"}
    population_size = 16  # @param {type:"integer"}
    optimizer = "adam"  # @param {type:"string"}
    learning_rate = 0.01  # @param
    seed = 42  # @param {type:"integer"}
    generations = 20  # @param {type:"integer"}
    n_seeds = 1  # @param {type:"integer"}

    SEED = jax.random.PRNGKey(seed)

    ES_CONFIG = {
        "popsize": population_size,
        "opt_name": optimizer,
        "lrate_init": learning_rate,
    }

    nb_k = 20
    c0 = [0] * nb_k
    c1 = [[i for i in range(nb_k)]]
    fl_config = FL_Config(X=256, Y=256, C=1, k=nb_k, c0=c0, c1=c1, dt=0.2, dd=4)

    config = ExperimentConfig(
        name="directed_motion",
        task=EvoFlow_DirectedMotion,
        fl_config=fl_config,
        es_config=ES_CONFIG,
        seed=SEED,
        n_seeds=n_seeds,
        generations=generations,
    )

    make_expe(config, SAVE_DIR, save_rate=SAVE_RATE)
