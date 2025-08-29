# core/planner.py
class Planner:
    """
    Abstract base class for planners.
    """
    def __init__(self, sampling_method):
        self.sampling_method = sampling_method

    def plan(self, start, goal, cost_map, weights):
        raise NotImplementedError