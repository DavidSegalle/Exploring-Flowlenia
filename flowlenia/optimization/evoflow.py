from evosax import OpenES, ParameterReshaper, FitnessShaper

import jax
import jax.numpy as jnp
import chex
from functools import partial
import typing as t

from flowlenia.flowlenia_params import FlowLeniaParams as FlowLenia, Config as FL_Config
from flowlenia.utils import get_kernels_fft

from flowlenia.optimization.optutils import *
from flowlenia.utils import loop_tqdm, scan_tqdm


@chex.dataclass
class Params:
    """Summary"""

    r: jnp.ndarray
    b: jnp.ndarray
    w: jnp.ndarray
    a: jnp.ndarray
    m: jnp.ndarray
    s: jnp.ndarray
    h: jnp.ndarray
    R: float


def compile_kernel_computer(X: int, Y: int, nb_k: int):
    def compute_kernels(params: Params):
        return get_kernels_fft(
            X, Y, nb_k, params.R, params.r, params.a, params.w, params.b
        )

    return jax.jit(compute_kernels)


class Rule_space:
    """Rule space for Flow Lenia system

    Attributes:
        nb_k (int): number of kernels of the system
        spaces (TYPE): Description

    Deleted Attributes:
        init_shape (TYPE): Description
    """

    # -----------------------------------------------------------------------------
    def __init__(self, nb_k: int):
        """

        Args:
            nb_k (int): number of kernels in the update rule
        """
        self.nb_k = nb_k
        self.kernel_keys = "r b w a m s h".split()
        self.spaces = {
            "r": {"low": 0.2, "high": 1.0, "mut_std": 0.2, "shape": None},
            "b": {"low": 0.001, "high": 1.0, "mut_std": 0.2, "shape": (3,)},
            "w": {"low": 0.01, "high": 0.5, "mut_std": 0.2, "shape": (3,)},
            "a": {"low": 0.0, "high": 1.0, "mut_std": 0.2, "shape": (3,)},
            "m": {"low": 0.05, "high": 0.5, "mut_std": 0.2, "shape": None},
            "s": {"low": 0.001, "high": 0.18, "mut_std": 0.01, "shape": None},
            "h": {"low": 0.01, "high": 1.0, "mut_std": 0.2, "shape": None},
            "R": {"low": 2.0, "high": 25.0, "mut_std": 0.2, "shape": None},
        }

    # -----------------------------------------------------------------------------
    def sample(self, key: jnp.ndarray) -> Params:
        """sample a random set of parameters

        Returns:
            Params: sampled parameters
        """
        kernels = {}
        for k in "rmsh":
            key, subkey = jax.random.split(key)
            kernels[k] = jax.random.uniform(
                key=subkey,
                minval=self.spaces[k]["low"],
                maxval=self.spaces[k]["high"],
                shape=(self.nb_k,),
            )
        for k in "awb":
            key, subkey = jax.random.split(key)
            kernels[k] = jax.random.uniform(
                key=subkey,
                minval=self.spaces[k]["low"],
                maxval=self.spaces[k]["high"],
                shape=(self.nb_k, 3),
            )
        R = jax.random.uniform(
            key=key, minval=self.spaces["R"]["low"], maxval=self.spaces["R"]["high"]
        )
        return Params(R=R, **kernels)


@chex.dataclass
class EvoParams(Params):
    init: jnp.ndarray


class EvoFlow:
    """OpenES wrapper for training Flow Lenia system"""

    # ---------------------------------------------------------------------------

    def __init__(self, fl_config: FL_Config, es_config, es_params=None):

        self.fl_config = fl_config
        self.fl, self.params_shaper = self.init_fl()

        self.es_config = es_config
        self.es_params = es_params
        self.strategy, self.fitness_shaper = self.init_strategy()

        self.evaluate = self._build_evaluator()

        self.run_fn = self._build_runner()

    # ---------------------------------------------------------------------------

    def init_fl(self):

        fl = FlowLenia(self.fl_config, key=jax.random.PRNGKey(42))
        params = fl.rule_space.sample(jax.random.PRNGKey(42))
        params = EvoParams(init=jnp.zeros((40, 40, self.fl_config.C)), **params)
        params_shaper = ParameterReshaper(params)

        return fl, params_shaper

    # ---------------------------------------------------------------------------

    def init_strategy(self):

        strategy = OpenES(num_dims=self.params_shaper.total_params, **self.es_config)
        if self.es_params is None:
            self.es_params = strategy.default_params
        fitness_shaper = FitnessShaper(maximize=True)
        return strategy, fitness_shaper

    # ---------------------------------------------------------------------------

    def _build_evaluator(self) -> t.Callable[[EvoParams, jnp.ndarray], jnp.ndarray]:

        raise NotImplementedError

    # ---------------------------------------------------------------------------

    def _build_runner(self):

        ask = jax.jit(self.strategy.ask)
        tell = jax.jit(self.strategy.tell)

        mu_params = EvoParams(
            init=0.5,
            **{
                k: (inf["low"] + inf["high"]) / 2
                for k, inf in self.fl.rule_space.spaces.items()
            },
        )
        lower_params = EvoParams(
            init=0.0, **{k: inf["low"] for k, inf in self.fl.rule_space.spaces.items()}
        )
        upper_params = EvoParams(
            init=jnp.inf,
            **{k: inf["high"] for k, inf in self.fl.rule_space.spaces.items()},
        )

        def get_init_state(key):
            init_state = self.strategy.initialize(key, self.es_params)
            return init_state

        @jax.jit
        def process_params(x):
            params = add_trees(self.params_shaper.reshape(x), mu_params)
            return clip_tree(params, lower_params, upper_params)

        @partial(jax.jit, static_argnums=(3, 4, 5))
        def _run(
            rng: jnp.ndarray,
            state,
            steps: jnp.ndarray,
            chkpt_file: str,
            save_rate: int,
            name="",
        ):

            @plot_scan(
                save_file=f"{chkpt_file}_plot.png",
                plot_rate=5,
                title=name,
                xlabel="generation",
                ylabel="fitness",
                color="k",
                alpha=0.2,
            )
            @save_scan(
                chkpt_file=chkpt_file,
                save_rate=save_rate,
                aux_infos={"fl_config": self.fl_config, "es_config": self.es_config},
            )
            @scan_tqdm(len(steps), 1)
            def es_step(carry, gen):
                key, state = carry
                key, ask_key, eval_key = jax.random.split(key, 3)
                x, state = ask(ask_key, state, self.es_params)
                params = process_params(x)
                fitness = self.evaluate(params, eval_key)
                fitness_re = self.fitness_shaper.apply(x, fitness)
                state = tell(x, fitness_re, state, self.es_params)
                return [key, state], fitness

            _, hist = jax.lax.scan(es_step, [rng, state], steps)
            return hist

        def run(
            rng: jnp.ndarray,
            steps: int,
            chkpt_file: str,
            save_rate: int,
            name="",
            chkpt=None,
        ):

            if chkpt is not None:
                key, state = chkpt["carry"]
                steps = jnp.arange(steps - state.gen_counter) + state.gen_counter
            else:
                key = rng
                state = get_init_state(rng)
                steps = jnp.arange(steps)
            return _run(key, state, steps, chkpt_file, save_rate, name)

        return run

    # ---------------------------------------------------------------------------
