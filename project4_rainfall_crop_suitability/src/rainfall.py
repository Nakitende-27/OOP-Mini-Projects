import numpy as np
MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

class Region:
    """Represent a region's monthly rainfall and summarise its pattern."""

    def __init__(self, name: str, rainfall: np.ndarray):
        rainfall = np.asarray(rainfall, dtype=float)

        if not name.strip():
            raise ValueError("Region name cannot be empty.")

        if rainfall.shape != (12,):
            raise ValueError("Rainfall must contain exactly 12 monthly values.")

        if not np.all(np.isfinite(rainfall)):
            raise ValueError("Rainfall values must be finite.")

        if np.any(rainfall < 0):
            raise ValueError("Rainfall cannot be negative.")

        self.name = name
        self.rainfall = rainfall

    def annual_total(self) -> float:
        """Return the total rainfall across the 12 months."""
        return float(np.sum(self.rainfall))

    def mean_rainfall(self) -> float:
        """Return the average monthly rainfall."""
        return float(np.mean(self.rainfall))

    def wettest_month(self) -> str:
        """Return the name of the month with the highest rainfall."""
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        """Return the name of the month with the lowest rainfall."""
        return MONTHS[int(np.argmin(self.rainfall))]

    def coefficient_of_variation(self) -> float:
        """Return the population coefficient of variation."""
        average = self.mean_rainfall()

        if average == 0:
            return float("nan")

        return float(np.std(self.rainfall, ddof=0) / average)

    def __repr__(self) -> str:
        """Return a concise representation of the region."""
        return (
            f"Region(name={self.name!r}, "
            f"months={len(self.rainfall)})"
        )

class CropRule:
    """Classify monthly rainfall against a crop's screening range."""

    def __init__(
        self,
        crop_name: str,
        min_rainfall: float,
        max_rainfall: float
    ):
        if not crop_name.strip():
            raise ValueError("Crop name cannot be empty.")

        if not np.isfinite(min_rainfall) or not np.isfinite(max_rainfall):
            raise ValueError("Rainfall thresholds must be finite.")

        if min_rainfall < 0:
            raise ValueError("Minimum rainfall cannot be negative.")

        if max_rainfall <= min_rainfall:
            raise ValueError(
                "Maximum rainfall must exceed minimum rainfall."
            )

        self.crop_name = crop_name
        self.min_rainfall = min_rainfall
        self.max_rainfall = max_rainfall

    def classify(self, rainfall: float) -> str:
        """Classify rainfall as drought risk, suitable, or excessive."""
        if not np.isfinite(rainfall) or rainfall < 0:
            raise ValueError("Rainfall must be finite and non-negative.")

        if rainfall < self.min_rainfall:
            return "Drought risk"

        if rainfall > self.max_rainfall:
            return "Waterlogging risk"

        return f"Good for {self.crop_name}"
