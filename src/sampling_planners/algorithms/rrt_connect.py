# algorithms/rrt_connect.py
from src.sampling_planners.core.planner import Planner
from src.sampling_planners.core.utils import collision_check, calculate_path_cost, nearest_neighbor
import numpy as np

class RRTConnect(Planner):
    """
    RRT-Connect planner.
    """
    def __init__(self, step_size=5.0, max_iterations=1000, sampling_method=None):
        super().__init__(sampling_method)
        self.step_size = step_size
        self.max_iterations = max_iterations

    def plan(self, start, goal, cost_map, weights):
        tree_a = [start]
        tree_b = [goal]
        parents_a = {start: None}
        parents_b = {goal: None}

        for _ in range(self.max_iterations):
            sample = self.sampling_method.sample(cost_map, start, goal)
            nearest_a = nearest_neighbor(tree_a, sample)
            new_a = self._steer(nearest_a, sample)
            if collision_check((nearest_a, new_a), cost_map):
                tree_a.append(new_a)
                parents_a[new_a] = nearest_a

                nearest_b = nearest_neighbor(tree_b, new_a)
                new_b = self._steer(nearest_b, new_a)
                if collision_check((nearest_b, new_b), cost_map):
                    tree_b.append(new_b)
                    parents_b[new_b] = nearest_b

                    if np.hypot(new_a[0] - new_b[0], new_a[1] - new_b[1]) < self.step_size:
                        path = self._trace_path(parents_a, new_a) + self._trace_path(parents_b, new_b)[::-1]
                        if calculate_path_cost(path, cost_map, weights) < float('inf'):
                            return path
        return None

    def _steer(self, from_pt, to_pt):
        direction = np.array(to_pt) - np.array(from_pt)
        dist = np.linalg.norm(direction)
        if dist < self.step_size:
            return tuple(to_pt)
        return tuple(np.array(from_pt) + direction / dist * self.step_size)

    def _trace_path(self, parents, node):
        path = [node]
        while parents[node] is not None:
            node = parents[node]
            path.append(node)
        return path[::-1]