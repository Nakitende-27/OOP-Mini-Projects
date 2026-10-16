import numpy as np
from abc import ABC, abstractmethod
class DistrictPopulation:

    def __init__(self, DistrictName, years, populations):
        self.DistrictName = DistrictName
        self.years = np.array(years)
        self.populations = np.array(populations)
        if len(self.years) != len(self.populations):
            raise ValueError("Years and populations must have equal lengths.")

        if len(self.years) == 0:
            raise ValueError("The data cannot be empty.")

        if np.any(self.populations < 0):
            raise ValueError("Population values cannot be negative.")
        if not np.all(np.isfinite(self.years)):
            raise ValueError("Years must be finite numbers.")

        if not np.all(self.years == self.years.astype(int)):
            raise ValueError("Years must be whole numbers.")

        if len(np.unique(self.years)) != len(self.years):
            raise ValueError("Years must be unique.")

        if not np.all(np.diff(self.years) > 0):
            raise ValueError("Years must be in ascending order.")

    def __len__(self) -> int:
        return len(self.populations)

    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self.DistrictName!r}, "
            f"years={len(self.years)}, "
            f"period={self.years[0]}-{self.years[-1]})"
        )

class Forecaster(ABC):
    """Abstract base class for population forecasting models."""

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, years, populations):
        """Fit the model to historical population data."""
        pass

    @abstractmethod
    def predict(self, horizon):
        """Predict population for the requested number of future years."""
        pass

class LinearTrendForecaster(Forecaster):
    """Forecast population using a fitted linear trend."""

    def fit(self, years, populations):
        years = np.asarray(years, dtype=float)
        populations = np.asarray(populations, dtype=float)

        if len(years) != len(populations) or len(years) < 2:
            raise ValueError("Provide at least two matching observations.")

        if not np.all(np.isfinite(years)) or not np.all(np.isfinite(populations)):
            raise ValueError("Inputs must be finite.")

        if np.any(populations < 0):
            raise ValueError("Populations cannot be negative.")

        self.coefficients = np.polyfit(years, populations, 1)
        self.last_year = years[-1]
        self.is_fitted = True
        return self

    def predict(self, horizon):
        if not self.is_fitted:
            raise RuntimeError("Fit the model before predicting.")

        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("Horizon must be a positive integer.")

        future_years = self.last_year + np.arange(1, horizon + 1)
        return np.polyval(self.coefficients, future_years)

class ExponentialForecaster(Forecaster):
    """Forecast population using constant compound annual growth."""

    def fit(self, years, populations):
        years = np.asarray(years, dtype=float)
        populations = np.asarray(populations, dtype=float)

        if len(years) != len(populations) or len(years) < 2:
            raise ValueError("Provide at least two matching observations.")

        if not np.all(np.isfinite(years)) or not np.all(np.isfinite(populations)):
            raise ValueError("Inputs must be finite.")

        if np.any(populations <= 0):
            raise ValueError("Exponential growth requires positive populations.")

        if not np.all(np.diff(years) > 0):
            raise ValueError("Years must be in ascending order.")

        intervals = years[-1] - years[0]
        self.rate = (populations[-1] / populations[0]) ** (1 / intervals) - 1
        self.last_population = populations[-1]
        self.last_year = years[-1]
        self.is_fitted = True
        return self

    def predict(self, horizon):
        if not self.is_fitted:
            raise RuntimeError("Fit the model before predicting.")

        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("Horizon must be a positive integer.")

        steps = np.arange(1, horizon + 1)
        return self.last_population * (1 + self.rate) ** steps

class FibonacciRatioForecaster(Forecaster):
    """Forecast by compounding successive Fibonacci-number ratios.

    This is an experimental benchmark, not an assumption that population
    genuinely grows according to Fibonacci ratios.
    """
    def fit(self, years, populations):
        years = np.asarray(years, dtype=float)
        populations = np.asarray(populations, dtype=float)

        if len(years) != len(populations) or len(years) < 2:
            raise ValueError("Provide at least two matching observations.")
        if not np.all(np.isfinite(years)) or not np.all(np.isfinite(populations)):
            raise ValueError("Inputs must be finite.")
        if np.any(populations <= 0):
            raise ValueError("Populations must be positive.")
        if not np.all(np.diff(years) > 0):
            raise ValueError("Years must be in ascending order.")

        # Fibonacci numbers chosen after the initial small ratios
        self.fibonacci = [13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
        self.ratios = [
            self.fibonacci[i + 1] / self.fibonacci[i]
            for i in range(len(self.fibonacci) - 1)
        ]

        self.last_population = populations[-1]
        self.last_year = years[-1]
        self.is_fitted = True
        return self

    def predict(self, horizon):
        if not self.is_fitted:
            raise RuntimeError("Fit the model before predicting.")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("Horizon must be a positive integer.")
        if horizon > len(self.ratios):
            raise ValueError("Horizon exceeds the available Fibonacci ratios.")

        predictions = []
        population = self.last_population

        for ratio in self.ratios[:horizon]:
            population = population * ratio
            predictions.append(population)

        return np.array(predictions)