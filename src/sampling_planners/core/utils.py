# core/utils.py
import numpy as np
from scipy.spatial import KDTree

def collision_check(line, cost_map, threshold=0.99, step=1.0):
    """
    Checks if the line segment is collision-free.
    Parameters
    ----------
    line : tuple
        ((x0, y0), (x1, y1)) endpoints.
    cost_map : np.ndarray
        2D cost map.
    threshold : float
        Maximum allowed cost value.
    step : float
        Step size for interpolation.
    Returns
    -------
    bool
        True if collision-free, False otherwise.
    """
    x0, y0 = line[0]
    x1, y1 = line[1]
    dist = np.hypot(x1 - x0, y1 - y0)
    num = int(dist / step) + 1
    xs = np.linspace(x0, x1, num)
    ys = np.linspace(y0, y1, num)
    for x, y in zip(xs, ys):
        xi, yi = int(round(x)), int(round(y))
        if xi < 0 or yi < 0 or xi >= cost_map.shape[0] or yi >= cost_map.shape[1]:
            return False
        if cost_map[xi, yi] >= threshold:
            return False
    return True

def calculate_path_cost(path, cost_map, weights):
    """
    Calculates weighted path cost.
    Parameters
    ----------
    path : list of tuple
        Sequence of (x, y) points.
    cost_map : np.ndarray
        2D cost map.
    weights : dict
        {'length': float, 'cost': float}
    Returns
    -------
    float
        Weighted cost J.
    """
    length = 0.0
    cost = 0.0
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i+1]
        length += np.hypot(p1[0] - p0[0], p1[1] - p0[1])
        cost += cost_map[int(round(p0[0])), int(round(p0[1]))]
    cost += cost_map[int(round(path[-1][0])), int(round(path[-1][1]))]
    return weights['length'] * length + weights['cost'] * cost

def heuristic(node, goal):
    """
    Euclidean distance heuristic.
    """
    return np.hypot(goal[0] - node[0], goal[1] - node[1])

def nearest_neighbor(nodes, point):
    """
    Finds nearest node using KDTree.
    Parameters
    ----------
    nodes : list of tuple
        Node coordinates.
    point : tuple
        Query point.
    Returns
    -------
    tuple
        Nearest node.
    """
    tree = KDTree(nodes)
    idx = tree.query(point)[1]
    return nodes[idx]