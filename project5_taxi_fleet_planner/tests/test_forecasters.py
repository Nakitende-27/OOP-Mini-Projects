
import numpy as np
import pytest

from src.forecasters import (
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    LinearTrendForecaster,
)


def test_moving_average_forecast():
    model = MovingAverageForecaster()
    history = np.array([10, 20, 30, 40])

    assert model.forecast(history) == 30


def test_exponential_smoothing_forecast():
    model = ExponentialSmoothingForecaster(alpha=0.5)
    history = np.array([10, 20, 30])

    assert model.forecast(history) == 22.5


def test_linear_trend_forecast():
    model = LinearTrendForecaster()
    history = np.array([10, 20, 30])

    assert model.forecast(history) == pytest.approx(40)


def test_exponential_smoothing_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        ExponentialSmoothingForecaster(alpha=0)


def test_moving_average_requires_three_observations():
    model = MovingAverageForecaster()

    with pytest.raises(ValueError):
        model.forecast(np.array([10, 20]))


def test_linear_trend_requires_two_observations():
    model = LinearTrendForecaster()

    with pytest.raises(ValueError):
        model.forecast(np.array([10]))
