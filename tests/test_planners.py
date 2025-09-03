# tests/test_planners.py
import sys
import os
from os.path import dirname, join, abspath
path = dirname(dirname(abspath(__file__)))
sys.path.append(path)
print(os.path)
import numpy as np
import pytest
from src.sampling_planners.core.sampling import UniformSampling, HybridSampling, StaticSampling, LearningBasedSampling
from src.sampling_planners.algorithms.rrt_connect import RRTConnect
from src.sampling_planners.algorithms.bidirectional_est import BidirectionalEST
from src.sampling_planners.algorithms.bit_star import BITStar
from src.sampling_planners.algorithms.prm import PRM

@pytest.fixture
def cost_map():
    np.random.seed(42)
    cmap = np.random.rand(128, 128)
    cmap[cmap > 0.8] = 1.0
    return cmap

@pytest.fixture
def start_goal():
    return (5, 5), (45, 45)

def test_rrt_connect(cost_map, start_goal):
    start, goal = start_goal
    planner = RRTConnect(step_size=5.0, max_iterations=500, sampling_method=LearningBasedSampling())
    path = planner.plan(start, goal, cost_map, {'length': 0.7, 'cost': 0.3})
    assert path is None or (path[0] == start and path[-1] == goal)

def test_bidirectional_est(cost_map, start_goal):
    start, goal = start_goal
    planner = BidirectionalEST(step_size=5.0, max_iterations=500, sampling_method=LearningBasedSampling())
    path = planner.plan(start, goal, cost_map, {'length': 0.7, 'cost': 0.3})
    assert path is None or (path[0] == start and path[-1] == goal)

def test_bit_star(cost_map, start_goal):
    start, goal = start_goal
    planner = BITStar(batch_size=50, max_batches=5, sampling_method=LearningBasedSampling())
    path = planner.plan(start, goal, cost_map, {'length': 0.7, 'cost': 0.3})
    assert path is None or (path[0] == start and path[-1] == goal)

def test_prm(cost_map, start_goal):
    start, goal = start_goal
    planner = PRM(n_samples=100, k_neighbors=5, sampling_method=LearningBasedSampling())
    path = planner.plan(start, goal, cost_map, {'length': 0.7, 'cost': 0.3})
    assert path is None or (path[0] == start and path[-1] == goal)
