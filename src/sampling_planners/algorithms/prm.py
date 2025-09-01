# algorithms/prm.py
from src.sampling_planners.core.planner import Planner
from src.sampling_planners.core.utils import collision_check, calculate_path_cost, heuristic, nearest_neighbor
from scipy.spatial import KDTree
import numpy as np
import heapq

class PRM(Planner):
    """
    Probabilistic Roadmap Planner.
    """
    def __init__(self, n_samples=200, k_neighbors=10, sampling_method=None):
        super().__init__(sampling_method)
        self.n_samples = n_samples
        self.k_neighbors = k_neighbors

    def plan(self, start, goal, cost_map, weights):
        samples = [start, goal]
        while len(samples) < self.n_samples + 2:
            pt = self.sampling_method.sample(cost_map, start, goal)
            if cost_map[int(round(pt[0])), int(round(pt[1]))] < 0.99:
                samples.append(pt)
        tree = KDTree(samples)
        edges = {pt: [] for pt in samples}
        for i, pt in enumerate(samples):
            dists, idxs = tree.query(pt, k=self.k_neighbors + 1)
            for j in idxs[1:]:
                neighbor = samples[j]
                if collision_check((pt, neighbor), cost_map):
                    cost = weights['length'] * heuristic(pt, neighbor) + weights['cost'] * cost_map[int(round(pt[0])), int(round(pt[1]))]
                    edges[pt].append((neighbor, cost))
        # A* search
        path, cost = self._astar(start, goal, edges, weights)
        return path

    def _astar(self, start, goal, edges, weights):
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start, [start]))
        closed = set()
        while open_set:
            f, g, node, path = heapq.heappop(open_set)
            if node == goal:
                return path, g
            if node in closed:
                continue
            closed.add(node)
            for neighbor, cost in edges.get(node, []):
                if neighbor in closed:
                    continue
                g_new = g + cost
                f_new = g_new + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_new, g_new, neighbor, path + [neighbor]))
        return None, float('inf')