
from abc import ABC, abstractmethod

import numpy as np


class Forecaster(ABC):
    """Base class for passenger-demand forecasting methods."""

    @abstractmethod
    def forecast(self, history: np.ndarray) -> float:
        """Forecast the next observation using historical values."""
        raise NotImplementedError


class MovingAverageForecaster(Forecaster):
    """Forecast using the mean of the most recent three observations."""

    def forecast(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)

        if history.ndim != 1 or len(history) < 3:
            raise ValueError("At least three observations are required.")

        if not np.all(np.isfinite(history)):
            raise ValueError("History must contain finite values.")

        return float(np.mean(history[-3:]))


class ExponentialSmoothingForecaster(Forecaster):
    """Forecast using simple exponential smoothing."""

    def __init__(self, alpha: float = 0.3):
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("Alpha must be greater than 0 and at most 1.")

        self.alpha = alpha

    def forecast(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)

        if history.ndim != 1 or len(history) < 1:
            raise ValueError("At least one observation is required.")

        if not np.all(np.isfinite(history)):
            raise ValueError("History must contain finite values.")

        level = float(history[0])

        for observation in history[1:]:
            level = (
                self.alpha * observation
                + (1 - self.alpha) * level
            )

        return float(level)


class LinearTrendForecaster(Forecaster):
    """Forecast the next observation using a fitted linear trend."""

    def forecast(self, history: np.ndarray) -> float:
        history = np.asarray(history, dtype=float)

        if history.ndim != 1 or len(history) < 2:
            raise ValueError("At least two observations are required.")

        if not np.all(np.isfinite(history)):
            raise ValueError("History must contain finite values.")

        days = np.arange(1, len(history) + 1)

        slope, intercept = np.polyfit(days, history, 1)

        return float(slope * (len(history) + 1) + intercept)
