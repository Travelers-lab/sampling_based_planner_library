# algorithms/bidirectional_est.py
from src.sampling_planners.core.planner import Planner
from src.sampling_planners.core.utils import collision_check, calculate_path_cost, nearest_neighbor
import numpy as np

class BidirectionalEST(Planner):
    """
    Bidirectional Expansive Space Trees.
    """
    def __init__(self, step_size=5.0, max_iterations=1000, k_samples=10, radius=10.0, sampling_method=None):
        super().__init__(sampling_method)
        self.step_size = step_size
        self.max_iterations = max_iterations
        self.k_samples = k_samples
        self.radius = radius

    def plan(self, start, goal, cost_map, weights):
        tree_a = [start]
        tree_b = [goal]
        parents_a = {start: None}
        parents_b = {goal: None}

        for _ in range(self.max_iterations):
            # Expand tree_a
            q = tree_a[np.random.randint(len(tree_a))]
            best_new = None
            best_cost = float('inf')
            for _ in range(self.k_samples):
                theta = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(0, self.radius)
                sample = (q[0] + r * np.cos(theta), q[1] + r * np.sin(theta))
                new = self._steer(q, sample)
                if collision_check((q, new), cost_map):
                    path = self._trace_path(parents_a, q) + [new]
                    cost = calculate_path_cost(path, cost_map, weights)
                    if cost < best_cost:
                        best_new = new
                        best_cost = cost
            if best_new is None:
                tree_a, tree_b = tree_b, tree_a
                parents_a, parents_b = parents_b, parents_a
                continue

            # Try to connect to tree_b
            nearest_b = nearest_neighbor(tree_b, best_new)
            if np.hypot(best_new[0] - nearest_b[0], best_new[1] - nearest_b[1]) < self.step_size:
                if collision_check((best_new, nearest_b), cost_map):
                    path = self._trace_path(parents_a, best_new) + self._trace_path(parents_b, nearest_b)[::-1]
                    if calculate_path_cost(path, cost_map, weights) < float('inf'):
                        return path

            # Swap trees
            tree_a, tree_b = tree_b, tree_a
            parents_a, parents_b = parents_b, parents_a
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