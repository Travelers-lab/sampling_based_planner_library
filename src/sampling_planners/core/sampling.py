# core/sampling.py
from os.path import dirname, join, abspath
import numpy as np
from scipy.ndimage import distance_transform_edt
from src.model.predictor import Predictor

class SamplingMethod:
    """Base class for sampling methods."""
    def sample(self, cost_map, start, goal):
        raise NotImplementedError

class UniformSampling(SamplingMethod):
    """Uniform random sampling."""
    def sample(self, cost_map, start, goal):
        h, w = cost_map.shape
        return (np.random.randint(0, h), np.random.randint(0, w))

class HybridSampling(SamplingMethod):
    """Uniform + goal-biased sampling."""
    def __init__(self, goal, goal_bias=0.1):
        self.goal = goal
        self.goal_bias = goal_bias

    def sample(self, cost_map, start, goal):
        if np.random.rand() < self.goal_bias:
            return self.goal
        h, w = cost_map.shape
        return (np.random.randint(0, h), np.random.randint(0, w))

class StaticSampling(SamplingMethod):
    """Static, user-provided samples."""
    def __init__(self, points):
        self.points = points
        self.idx = 0

    def sample(self, cost_map, start, goal):
        pt = self.points[self.idx]
        self.idx = (self.idx + 1) % len(self.points)
        return pt

class LearningBasedSampler(SamplingMethod):
    
    def __init__(self, goal_bias=0.1, device='cpu'):
        self.goal_bias = goal_bias
        self.predictor = Predictor(join(dirname(dirname(dirname(abspath(__file__)))),"model/best_model.pth"), device)
        
    def sample(self, cost_map, start, goal):
        if np.random.rand() < self.goal_bias:
            return tuple(goal)
        obstacle_map, start_map, goal_map, distance_map = self._generate_feature_maps(cost_map, start, goal)
        prob_map = self.predictor.predict(obstacle_map, start_map, goal_map, distance_map)
        sampled_point = self._sample_from_probability(prob_map, cost_map)
        
        return sampled_point
    
    def _generate_feature_maps(self, cost_map, start, goal):
        h, w = cost_map.shape
        obstacle_map = (cost_map == 1).astype(np.float32)
        start_map = np.zeros_like(cost_map, dtype=np.float32)
        start_row, start_col = start
        start_map[start_row, start_col] = 1.0
        goal_map = np.zeros_like(cost_map, dtype=np.float32)
        goal_row, goal_col = goal
        goal_map[goal_row, goal_col] = 1.0
        free_space = 1 - obstacle_map
        distance_map = distance_transform_edt(free_space)
        if np.max(distance_map) > 0:
            distance_map = distance_map / np.max(distance_map)
        
        return obstacle_map, start_map, goal_map, distance_map
    
    def _sample_from_probability(self, prob_map, cost_map):
        h, w = prob_map.shape
        prob_map_flat = prob_map.flatten()
        prob_map_flat = np.maximum(prob_map_flat, 0) 
        prob_map_flat = prob_map_flat / np.sum(prob_map_flat) 
        sampled_idx = np.random.choice(len(prob_map_flat), p=prob_map_flat)
        sampled_row = sampled_idx // w
        sampled_col = sampled_idx % w
        max_attempts = 100
        for attempt in range(max_attempts):
            if cost_map[sampled_row, sampled_col] < 0.9: 
                return (sampled_row, sampled_col)
            sampled_idx = np.random.choice(len(prob_map_flat), p=prob_map_flat)
            sampled_row = sampled_idx // w
            sampled_col = sampled_idx % w
        return self._get_random_free_point(cost_map)
    
    def _get_random_free_point(self, cost_map):
        h, w = cost_map.shape
        free_points = np.where(cost_map < 0.9) 
        
        if len(free_points[0]) > 0:
            idx = np.random.randint(len(free_points[0]))
            return (free_points[0][idx], free_points[1][idx])
        else:
            return (h // 2, w // 2)
    
    def set_goal_bias(self, goal_bias):
        self.goal_bias = max(0.0, min(1.0, goal_bias))