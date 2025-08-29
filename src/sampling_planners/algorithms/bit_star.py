# algorithms/bit_star.py
from sampling_planners.core.planner import Planner
from sampling_planners.core.utils import collision_check, calculate_path_cost, heuristic, nearest_neighbor
import numpy as np
import heapq

class BITStar(Planner):
    """
    Simplified BIT* planner.
    """
    def __init__(self, batch_size=100, max_batches=10, step_size=5.0, sampling_method=None):
        super().__init__(sampling_method)
        self.batch_size = batch_size
        self.max_batches = max_batches
        self.step_size = step_size

    def plan(self, start, goal, cost_map, weights):
        best_cost = float('inf')
        best_path = None
        samples = [start, goal]
        edges = {}
        for batch in range(self.max_batches):
            # Sample new batch in informed ellipse
            for _ in range(self.batch_size):
                sample = self.sampling_method.sample(cost_map)
                if heuristic(start, sample) + heuristic(sample, goal) < best_cost:
                    samples.append(sample)
            # Build edges
            for i, s1 in enumerate(samples):
                for j, s2 in enumerate(samples):
                    if i == j: continue
                    if heuristic(s1, s2) > self.step_size: continue
                    if collision_check((s1, s2), cost_map):
                        cost = weights['length'] * heuristic(s1, s2) + weights['cost'] * cost_map[int(round(s1[0])), int(round(s1[1]))]
                        edges.setdefault(s1, []).append((s2, cost))
            # A* search
            path, cost = self._astar(start, goal, edges, weights)
            if path and cost < best_cost:
                best_cost = cost
                best_path = path
        return best_path

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