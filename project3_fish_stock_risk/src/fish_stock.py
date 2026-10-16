import numpy as np
from statistics import mean, stdev
class FishStock:
    """Simulate fish stock using logistic growth with harvesting."""

    def __init__(
        self,
        initial_stock: float = 4000,
        growth_rate: float = 0.4,
        carrying_capacity: float = 10000,
        harvest_rate: float = 0.10
    ):
        """Initialise the fish stock model."""
        if initial_stock < 0:
            raise ValueError("Initial stock cannot be negative.")

        if growth_rate < 0:
            raise ValueError("Growth rate cannot be negative.")

        if carrying_capacity <= 0:
            raise ValueError("Carrying capacity must be positive.")

        if initial_stock > carrying_capacity:
            raise ValueError("Initial stock cannot exceed carrying capacity.")

        if not 0 <= harvest_rate <= 1:
            raise ValueError("Harvest rate must be between 0 and 1.")

        self.initial_stock = initial_stock
        self.growth_rate = growth_rate
        self.carrying_capacity = carrying_capacity
        self.harvest_rate = harvest_rate

    def simulate(self, weeks: int = 52) -> np.ndarray:
        """Return the fish stock for the initial period and each week."""
        if weeks < 1:
            raise ValueError("Weeks must be at least 1.")

        stock = np.zeros(weeks + 1)
        stock[0] = self.initial_stock

        for week in range(weeks):
            current_stock = stock[week]

            natural_growth = (
                self.growth_rate
                * current_stock
                * (1 - current_stock / self.carrying_capacity)
            )

            harvest = self.harvest_rate * current_stock

            next_stock = current_stock + natural_growth - harvest

            # Prevent negative stock in the simulation.
            stock[week + 1] = max(0, next_stock)

        return stock

class PriceModel:
    """Simulate weekly fish prices using a bounded random walk."""

    def __init__(
        self,
        initial_price: float = 12000,
        min_price: float = 9000,
        max_price: float = 16000,
        seed: int = 42
    ):
        """Initialise the price model."""
        if not min_price <= initial_price <= max_price:
            raise ValueError("Initial price must be within the price bounds.")

        if min_price <= 0 or max_price <= min_price:
            raise ValueError("Price bounds must be positive and ordered.")

        self.initial_price = initial_price
        self.min_price = min_price
        self.max_price = max_price
        self.seed = seed

    def simulate(self, weeks: int = 52) -> np.ndarray:
        """Simulate prices for the initial week and subsequent weeks."""
        if weeks < 1:
            raise ValueError("Weeks must be at least 1.")

        rng = np.random.default_rng(self.seed)
        prices = np.zeros(weeks + 1)
        prices[0] = self.initial_price

        for week in range(weeks):
            # Simulate a weekly price change in UGX/kg.
            change = rng.normal(loc=0, scale=400)

            prices[week + 1] = np.clip(
                prices[week] + change,
                self.min_price,
                self.max_price
            )

        return prices

class RiskAssessor:
    """Classify revenue risk using the coefficient of variation."""

    def __init__(
        self,
        low_threshold: float = 0.10,
        high_threshold: float = 0.25
    ):
        """Set the thresholds used for risk classification."""
        if not 0 <= low_threshold < high_threshold:
            raise ValueError("Risk thresholds must be ordered and non-negative.")

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def coefficient_of_variation(
        self,
        revenues: list[float]
    ) -> float:
        """Calculate sample standard deviation divided by mean revenue."""
        if len(revenues) < 2:
            raise ValueError("At least two revenue values are required.")

        average_revenue = mean(revenues)

        if average_revenue <= 0:
            raise ValueError("Mean revenue must be positive.")

        return stdev(revenues) / average_revenue

    def classify(self, cv: float) -> str:
        """Return a risk category based on the CV."""
        if not np.isfinite(cv) or cv < 0:
            raise ValueError("CV must be finite and non-negative.")

        if cv < self.low_threshold:
            return "Low"
        elif cv < self.high_threshold:
            return "Moderate"
        else:
            return "High"