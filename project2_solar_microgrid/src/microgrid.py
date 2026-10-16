import numpy as np
from scipy.linalg import solve
from scipy.optimize import nnls

class MicroGrid:
    """Model a solar-battery micro-grid using two demand constraints."""
    def __init__(self, matrix=None):
        if matrix is None:
            matrix = [[3, 2], [4, 1]]

        self.matrix = np.asarray(matrix, dtype=float)

        if self.matrix.shape != (2, 2):
            raise ValueError("The coefficient matrix must be 2x2.")

        if not np.all(np.isfinite(self.matrix)):
            raise ValueError("Matrix values must be finite.")

        self.determinant = np.linalg.det(self.matrix)
        self.condition_number = np.linalg.cond(self.matrix)

        if np.isclose(self.determinant, 0):
            raise ValueError(
                "The coefficient matrix is singular or nearly singular."
            )

    def solve_day(self, d1, d2):
        demands = np.asarray([d1, d2], dtype=float)

        if not np.all(np.isfinite(demands)):
            raise ValueError("Demand values must be finite.")

        if np.any(demands < 0):
            raise ValueError("Demand values cannot be negative.")

        usage = solve(self.matrix, demands)

        return {
            "solar_kwh": usage[0],
            "battery_kwh": usage[1],
            "feasible": bool(np.all(usage >= 0))
        }


def solve_with_nnls(grid, d1, d2):
    demands = np.asarray([d1, d2], dtype=float)

    if not np.all(np.isfinite(demands)):
        raise ValueError("Demand values must be finite.")

    if np.any(demands < 0):
        raise ValueError("Demand values cannot be negative.")

    exact_usage = solve(grid.matrix, demands)

    if np.all(exact_usage >= 0):
        adjusted_usage = exact_usage
        used_nnls = False
    else:
        adjusted_usage, _ = nnls(grid.matrix, demands)
        used_nnls = True

    residual = grid.matrix @ adjusted_usage - demands

    return {
        "solar_kwh": adjusted_usage[0],
        "battery_kwh": adjusted_usage[1],
        "feasible": bool(np.all(adjusted_usage >= 0)),
        "used_nnls": used_nnls,
        "residual_norm": np.linalg.norm(residual)
    }


class HybridMicroGrid(MicroGrid):

    def __init__(self):
        super().__init__()

        self.hybrid_matrix = np.array([
            [3, 2, 0],
            [4, 1, 0],
            [1, 1, 1]
        ], dtype=float)

        self.hybrid_determinant = np.linalg.det(self.hybrid_matrix)
        self.hybrid_condition_number = np.linalg.cond(
            self.hybrid_matrix
        )

        if np.isclose(self.hybrid_determinant, 0):
            raise ValueError("The hybrid matrix is singular.")

    def solve_day(self, d1, d2, d3):
        demands = np.asarray([d1, d2, d3], dtype=float)

        if not np.all(np.isfinite(demands)):
            raise ValueError("Demand values must be finite.")

        if np.any(demands < 0):
            raise ValueError("Demand values cannot be negative.")

        usage = solve(self.hybrid_matrix, demands)

        return {
            "solar_kwh": usage[0],
            "battery_kwh": usage[1],
            "diesel_kwh": usage[2],
            "feasible": bool(np.all(usage >= 0))
        }
