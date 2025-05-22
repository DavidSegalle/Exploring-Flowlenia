from flowlenia.flowlenia_params import (
    FlowLeniaParams,
    Rule_space,
    Params,
    Config,
    beam_mutation,
)
from flowlenia.environments import init_connected_cells
from flowlenia.environments import init_activation_params_grid, multi_cross_map

import chex
import numpy as np
from flowlenia.utils import conn_from_matrix

import jax
import jax.numpy as jnp
import jax.random as jr

from functools import partial

key = jr.PRNGKey(42)


@chex.dataclass
class MutationParams:
    mutation_rate: float
    beam_size: int


@chex.dataclass
class ExploState:
    params: Params
    config: Config
    mut_params: MutationParams


class ExploSpace:
    def __init__(self):
        self.M = np.array([[2, 1, 0], [0, 2, 1], [1, 0, 2]])
        self.k = self.M.sum()
        self.rs = Rule_space(self.k)

    def sample(self):
        params = self.rs.sample(key)
        cfg = Config(X=128, Y=128, C=3, k=self.k, mix_rule="negotiate", border="torus")
        c0, c1 = conn_from_matrix(self.M)
        cfg = cfg.replace(c0=c0, c1=c1)
        mut_params = MutationParams(mutation_rate=0.1, beam_size=5)
        return ExploState(params=params, config=cfg, mut_params=mut_params)

    def mutate(self, state: ExploState):
        new_params = Rule_space.mutate(state.params, key)
        new_config = state.config  # TODO: mutate config
        new_mut_params = state.mut_params
        new_mut_params = new_mut_params.replace(
            mutation_rate=jnp.clip(
                new_mut_params.mutation_rate + jr.normal(key) * 0.1, 0.0, 1.0
            ),
            beam_size=jnp.clip(new_mut_params.beam_size + jr.normal(key) * 1, 1, 10),
        )
        return ExploState(
            params=new_params, config=new_config, mut_params=new_mut_params
        )


@chex.dataclass
class Goal:
    ea: float
    size: int
    complexity: float


class GoalSpace:
    def sample(self):
        ea = jr.uniform(key, minval=0.0, maxval=1000.0)
        size = jr.randint(key, minval=0, maxval=100)
        complexity = jr.uniform(key, minval=0.0, maxval=1000.0)

    def get_closest(self, goal: Goal, explored: dict):
        closest_goal = None
        closest_params = None
        min_dist = float("inf")
        for params, state in explored.items():
            dist = self.distance_goal(goal, state)
            if dist < min_dist:
                min_dist = dist
                closest_goal = state
                closest_params = params
        return closest_goal, closest_params

    def distance_goal(self, goal1: Goal, goal2: Goal):
        return (
            abs(goal1.ea - goal2.ea)
            + abs(goal1.size - goal2.size)
            + abs(goal1.complexity - goal2.complexity)
        )


def run_imgep_explorer(n_iterations: int):
    explored_states = {}
    gs = GoalSpace()
    es = ExploSpace()
    flp = FlowLeniaParams()
    for i in range(n_iterations):
        goal = gs.sample()
        closest_goal, closest_params = gs.get_closest(goal, explored_states)
        if closest_goal is None:
            state = es.sample()
        else:
            state = es.mutate(closest_params)

        flp, init_state = setup_flp(state)
        reached_goal = run_simulation(flp, init_state)


def setup_flp(state: ExploState):
    callback = partial(beam_mutation, sz=20, p=0.1)
    flp = FlowLeniaParams(cfg=state.config, callback=callback)
    s = init_activation_params_grid(flp, seed=42, init_mask=state.params)
    return flp, s


def run_simulation(flp: FlowLeniaParams, init_state):
    pass  # TODO: implement


X = 256
Y = 256
C = 3
width = 8
cluster_padding = 8
n_clusters = 4

init_mask = multi_cross_map(X, Y, width + cluster_padding, 1.0, n=n_clusters)
