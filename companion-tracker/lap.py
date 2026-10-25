"""Stand-in for the `lap` package (only lapjv, used by the Ultralytics tracker), built on scipy.

Lets the app run when `pip install lap` isn't possible. If the real `lap` is installed later,
this file can stay: it gives the same results for the tracker.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

__version__ = "0.5.12-scipy"


def lapjv(cost, extend_cost=True, cost_limit=np.inf, return_cost=True):
    """Same contract as lap.lapjv: returns (total_cost, x, y) where x[i] is the column assigned
    to row i (-1 if none) and y[j] the row assigned to column j (-1 if none). Pairs costing more
    than cost_limit are left unassigned (the usual extended-matrix construction)."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    x = -np.ones(n, dtype=int)
    y = -np.ones(m, dtype=int)
    if n == 0 or m == 0:
        return (0.0, x, y) if return_cost else (x, y)
    if np.isfinite(cost_limit):
        big = cost_limit / 2.0
        ext = np.full((n + m, n + m), 0.0)
        ext[:n, :m] = cost
        ext[:n, m:] = np.inf
        ext[n:, :m] = np.inf
        np.fill_diagonal(ext[:n, m:], big)     # row i left unmatched
        np.fill_diagonal(ext[n:, :m], big)     # column j left unmatched
        ext[~np.isfinite(ext)] = 1e18
        rows, cols = linear_sum_assignment(ext)
        for r, c in zip(rows, cols):
            if r < n and c < m:
                x[r], y[c] = c, r
    else:
        rows, cols = linear_sum_assignment(cost)
        x[rows], y[cols] = cols, rows
    total = float(sum(cost[i, x[i]] for i in range(n) if x[i] >= 0))
    return (total, x, y) if return_cost else (x, y)
