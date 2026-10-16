
import numpy as np
import pytest

from src.taxi_route import Route


def test_route_revenue_and_statistics():
    route = Route("Test Route", np.array([10, 20, 30]), 2000)

    assert route.daily_revenue().tolist() == [
        20000, 40000, 60000
    ]
    assert route.total_revenue() == 120000
    assert route.mean_passengers() == 20
    assert route.variance_passengers() == 100
    assert route.standard_deviation() == pytest.approx(10)


def test_route_rejects_negative_passengers():
    with pytest.raises(ValueError):
        Route("Test Route", np.array([10, -2, 20]), 2000)


def test_route_rejects_non_integer_passengers():
    with pytest.raises(ValueError):
        Route("Test Route", np.array([10, 2.5, 20]), 2000)


def test_route_rejects_invalid_fare():
    with pytest.raises(ValueError):
        Route("Test Route", np.array([10, 20, 30]), 0)


def test_route_requires_at_least_two_days():
    with pytest.raises(ValueError):
        Route("Test Route", np.array([10]), 2000)
