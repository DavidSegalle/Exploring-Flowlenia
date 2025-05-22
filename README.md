# Exploring-Flowlenia

Exploring-Flowlenia is a framework for simulating, optimizing, and analyzing Flow-Lenia cellular automata systems. Flow-Lenia is an extension of the Lenia cellular automaton model that incorporates fluid dynamics principles to create more complex and emergent behaviors.

## Overview

This project provides tools for:
- Creating and simulating Flow-Lenia systems with different parameters
- Optimizing Flow-Lenia parameters using evolutionary algorithms
- Analyzing the complexity and behaviors of Flow-Lenia patterns
- Automatic discovery of interesting patterns using techniques like IMGEP (Intrinsically Motivated Goal Exploration Processes)

## Requirements

This project uses python 3.10.*

## Installation

To begin, set up a virtual environment and install the required packages. Then install the flowlenia package in editable mode. You can do this by running the following commands:

```bash
pip install -r requirements.txt
pip install -e .
```

By default, Jax is installed with the cpu version. To install the gpu version, follow the instructions at [https://jax.readthedocs.io/en/latest/installation.html](https://jax.readthedocs.io/en/latest/installation.html)

## Core Components

The project consists of the following main components:

### FlowLenia Simulation
- `flowlenia_vanilla.py` - The core implementation of the Flow-Lenia model
- `flowlenia_params.py` - Variants of the Flow-Lenia model with parameter localization

### Visualization and Utilities
- `vizutils.py` - Visualization tools for Flow-Lenia simulations
- `simutils.py` - Simulation utilities
- `utils.py` - General utility functions

### Optimization
- `optimization/evoflow.py` - Evolutionary algorithms for optimizing Flow-Lenia parameters
- `optimization/tasks.py` - Task definitions for evolutionary optimization
- `optimization/optexp.py` - Experiment configuration for optimization runs

### Analysis
- `analysis/` - Tools for analyzing Flow-Lenia patterns and behaviors

### Auto-discovery
- `autodisc/` - Automatic discovery of interesting patterns using IMGEP

## Example Usage

See the `examples/` directory for examples of how to use the Flow-Lenia framework.
Here's a simple example of creating and simulating a Flow-Lenia system:

```python
from flowlenia.flowlenia_vanilla import FlowLenia, Config, State
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from flowlenia.utils import conn_from_matrix

# Set system configuration
M = np.array([[3, 1, 0], [0, 3, 1], [1, 0, 3]], dtype=int)
k = M.sum()
c0, c1 = conn_from_matrix(M)
cfg = Config(X=128, Y=128, C=3, k=k, c0=c0, c1=c1)

# Initialize system
key = jr.key(1111)
fl = FlowLenia(cfg, key=key)

# Initialize state
s = fl.initialize()
A = s.A.at[44:84, 44:84, :].set(jr.uniform(key, (40, 40, cfg.C)))
s = s.replace(A=A)

# Simulate rollout
s, ss = fl.rollout(s, steps=100)
```

## Experiments

The `experiments/` directory contains various experiments showcasing different aspects of Flow-Lenia:

- `complexity_analysis/` - Analyzing the complexity of Flow-Lenia patterns
- `distribution/` - Experiments with diffusion of matter
- `ea_walls/` - Evolutionary activity with wall constraints
- `large_map/` - Simulations with large grid sizes
- `metamorph/` - Experiments with metamorphosing patterns
- `mixing/` - Parameter mixing experiments

