import jax
import jax.numpy as jnp
import jax.random as jr
from functools import partial
from flowlenia.optimization.optutils import center_of_mass
from flowlenia.optimization.evoflow import EvoFlow, EvoParams
from flowlenia.flowlenia_params import State as FL_State


class EvoFlow_DirectedMotion(EvoFlow):

    def _build_evaluator(self):
        ker_comp_map = jax.vmap(self.fl.kernel_computer)
        com = partial(center_of_mass, X=self.fl_config.X, Y=self.fl_config.Y)
        com = jax.vmap(com)
        X, Y, C = self.fl_config.X, self.fl_config.Y, self.fl_config.C
        mx, my = X // 2, Y // 2
        popsize = self.es_config["popsize"]

        rollout_fn = jax.vmap(lambda s, k: self.fl.rollout(s, k, 400)[0])

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
            coms = com(jnp.sum(finals.A, axis=-1, keepdims=True))
            return jnp.sqrt(jnp.sum(coms**2, axis=-1))

        return evaluate
