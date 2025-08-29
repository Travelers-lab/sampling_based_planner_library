# core/sampling.py
import numpy as np

class SamplingMethod:
    """Base class for sampling methods."""
    def sample(self, cost_map):
        raise NotImplementedError

class UniformSampling(SamplingMethod):
    """Uniform random sampling."""
    def sample(self, cost_map):
        h, w = cost_map.shape
        return (np.random.randint(0, h), np.random.randint(0, w))

class HybridSampling(SamplingMethod):
    """Uniform + goal-biased sampling."""
    def __init__(self, goal, goal_bias=0.1):
        self.goal = goal
        self.goal_bias = goal_bias

    def sample(self, cost_map):
        if np.random.rand() < self.goal_bias:
            return self.goal
        h, w = cost_map.shape
        return (np.random.randint(0, h), np.random.randint(0, w))

class StaticSampling(SamplingMethod):
    """Static, user-provided samples."""
    def __init__(self, points):
        self.points = points
        self.idx = 0

    def sample(self, cost_map):
        pt = self.points[self.idx]
        self.idx = (self.idx + 1) % len(self.points)
        return pt