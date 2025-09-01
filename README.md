# sampling_planners

A Python library for sampling-based path planning with cost maps.

## Features

- RRT-Connect, Bidirectional EST, BIT*, PRM algorithms
- Weighted cost map support
- Multiple sampling strategies
- Modular, extensible design

## Usage Guide
**Installation**
You can install `sampling_planners` using pip:
```bash
git clone git@github.com:Travelers-lab/sampling_based_planner_library.git
cd sampling_planners
pip install .
```
You can install `sampling_planners` using pip:
```bash
pip install -e .
```
### Basic Usage Example
Here's a simple example demonstrating how to plan a path using RRT-Connect with uniform sampling:
```bash
import numpy as np
from sampling_planners import RRTConnect, UniformSampling

# Create a random cost map (100x100 grid)
cost_map = np.random.rand(128, 128)
cost_map[cost_map > 0.8] = 1.0  

start = (10, 10)
goal = (90, 90)

planner = RRTConnect(step_size=5.0, max_iterations=1000, sampling_method=UniformSampling())
path = planner.plan(start, goal, cost_map, {'length': 0.7, 'cost': 0.3})

print("Path found:", path)
```
### Advanced Examples
1. Using Hybrid Sampling with Bidirectional EST
```bash
from sampling_planners import BidirectionalEST, HybridSampling

cost_map = np.random.rand(100, 100)
cost_map[cost_map > 0.8] = 1.0

start = (5, 5)
goal = (95, 95)

sampling = HybridSampling(goal=goal, goal_bias=0.2)
planner = BidirectionalEST(step_size=4.0, max_iterations=800, sampling_method=sampling)
path = planner.plan(start, goal, cost_map, {'length': 0.5, 'cost': 0.5})
```
2. Planning with PRM and Static Sampling
```bash
from sampling_planners import PRM, StaticSampling

cost_map = np.random.rand(50, 50)
cost_map[cost_map > 0.85] = 1.0

start = (2, 2)
goal = (48, 48)
static_points = [(x, y) for x in range(0, 50, 5) for y in range(0, 50, 5)]

sampling = StaticSampling(points=static_points)
planner = PRM(n_samples=100, k_neighbors=8, sampling_method=sampling)
path = planner.plan(start, goal, cost_map, {'length': 0.6, 'cost': 0.4})
```
3. BIT* with Custom Weights
```bash
from sampling_planners import BITStar, UniformSampling

cost_map = np.random.rand(80, 80)
cost_map[cost_map > 0.9] = 1.0

start = (10, 10)
goal = (70, 70)

planner = BITStar(batch_size=50, max_batches=5, step_size=6.0, sampling_method=UniformSampling())
path = planner.plan(start, goal, cost_map, {'length': 0.8, 'cost': 0.2})
```
## Common Pitfalls
Start/Goal in Obstacles: Ensure start and goal are in low-cost, operable regions (cost_map < 1.0).
Cost Map Shape: The cost map must be a 2D numpy array; mismatched shapes may cause errors.
Sampling Method: Use an appropriate sampling method for your scenario; static sampling requires a non-empty list of points.
Parameter Tuning: Adjust step_size, max_iterations, and weights for best results on your map.
Path Not Found: If no path is found, try increasing max_iterations or adjusting the cost map and weights.
## API Reference
See the [API Documentation]() for detailed class and method descriptions.
## License
License Type: MIT License
Copyright Notice:
Copyright (c) 2024 Travelers-lab

Permissions:
 - Modification
 - Distribution
 - Private use

For business cooperation inquiries, please contact 2210991@tongji.edu.cn.