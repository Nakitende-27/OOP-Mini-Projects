
import numpy as np
from statistics import mean, variance, stdev

class Route:
    """Represent passenger demand and fare revenue for a taxi route."""

    def __init__(
        self,
        name: str,
        passenger_counts: np.ndarray,
        fare: float
    ):
        passenger_counts = np.asarray(passenger_counts, dtype=float)

        if not name.strip():
            raise ValueError("Route name cannot be empty.")

        if passenger_counts.ndim != 1 or len(passenger_counts) < 2:
            raise ValueError("Provide at least two daily passenger counts.")

        if not np.all(np.isfinite(passenger_counts)):
            raise ValueError("Passenger counts must be finite.")

        if np.any(passenger_counts < 0):
            raise ValueError("Passenger counts cannot be negative.")

        if not np.all(passenger_counts == np.floor(passenger_counts)):
            raise ValueError("Passenger counts must be whole numbers.")

        if not np.isfinite(fare) or fare <= 0:
            raise ValueError("Fare must be a positive finite number.")

        self.name = name
        self.passenger_counts = passenger_counts.astype(int)
        self.fare = float(fare)

    def daily_revenue(self) -> np.ndarray:
        """Return the revenue for each observed day."""
        return self.passenger_counts * self.fare

    def total_revenue(self) -> float:
        """Return the total revenue across all observed days."""
        return float(np.sum(self.daily_revenue()))

    def mean_passengers(self) -> float:
        """Return the mean daily passenger count."""
        return mean(self.passenger_counts.tolist())

    def variance_passengers(self) -> float:
        """Return the sample variance of daily passenger counts."""
        return variance(self.passenger_counts.tolist())

    def standard_deviation(self) -> float:
        """Return the sample standard deviation of passenger counts."""
        return stdev(self.passenger_counts.tolist())

    def __repr__(self) -> str:
        """Return a concise representation of the route."""
        return (
            f"Route(name={self.name!r}, "
            f"days={len(self.passenger_counts)}, "
            f"fare={self.fare:.0f})"
        )
