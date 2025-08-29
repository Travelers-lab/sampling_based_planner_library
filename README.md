# sampling_planners

A Python library for sampling-based path planning with cost maps.

## Features

- RRT-Connect, Bidirectional EST, BIT*, PRM algorithms
- Weighted cost map support
- Multiple sampling strategies
- Modular, extensible design

## Installation
```bash
git clone git@github.com:Travelers-lab/sampling_based_planner_library.git
cd sampling_planners
pip install .
```

## Example Usage
```python
import numpy as np
from sampling_planners import RRTConnect, UniformSampling

cost_map = np.random.rand(100, 100)
cost_map[cost_map > 0.8] = 1.0
start, goal = (10, 10), (90, 90)
planner = RRTConnect(step_size=5.0, max_iterations=1000, sampling_method=UniformSampling())
path = planner.plan(start, goal, cost_map, {'length': 0.7, 'cost': 0.3})
```
