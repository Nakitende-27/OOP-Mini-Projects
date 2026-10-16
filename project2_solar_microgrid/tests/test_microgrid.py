"""Automated tests for the micro-grid dispatch module."""

import numpy as np
import pytest

from src.microgrid import MicroGrid, HybridMicroGrid, solve_with_nnls


def test_microgrid_solves_feasible_demands():
    """Check the solution for a feasible demand pair."""
    grid = MicroGrid()
    result = grid.solve_day(30, 20)

    assert result["solar_kwh"] == pytest.approx(2.0)
    assert result["battery_kwh"] == pytest.approx(12.0)
    assert result["feasible"] is True


def test_microgrid_rejects_negative_demand():
    """Negative demands should raise a ValueError."""
    grid = MicroGrid()

    with pytest.raises(ValueError):
        grid.solve_day(-10, 20)


def test_microgrid_rejects_non_finite_demand():
    """NaN and infinity should be rejected."""
    grid = MicroGrid()

    with pytest.raises(ValueError):
        grid.solve_day(np.nan, 20)

    with pytest.raises(ValueError):
        grid.solve_day(30, np.inf)


def test_nnls_handles_infeasible_exact_solution():
    """NNLS should return non-negative usage for incompatible demands."""
    grid = MicroGrid()
    result = solve_with_nnls(grid, 10, 100)

    assert result["used_nnls"] is True
    assert result["solar_kwh"] >= 0
    assert result["battery_kwh"] >= 0
    assert result["residual_norm"] > 0


def test_hybrid_microgrid_solves_three_constraints():
    """Check solar, battery and diesel results."""
    grid = HybridMicroGrid()
    result = grid.solve_day(30, 20, 25)

    assert result["solar_kwh"] == pytest.approx(2.0)
    assert result["battery_kwh"] == pytest.approx(12.0)
    assert result["diesel_kwh"] == pytest.approx(11.0)
    assert result["feasible"] is True


def test_hybrid_microgrid_rejects_negative_demand():
    """The hybrid model should reject negative demand."""
    grid = HybridMicroGrid()

    with pytest.raises(ValueError):
        grid.solve_day(30, 20, -5)
