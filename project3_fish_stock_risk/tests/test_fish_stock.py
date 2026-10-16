import numpy as np
import pytest

from src.fish_stock import FishStock, PriceModel, RiskAssessor


def test_fish_stock_simulation():
    """Stock simulation should return valid weekly stock values."""
    model = FishStock()
    stock = model.simulate(weeks=52)

    assert len(stock) == 53
    assert stock[0] == 4000
    assert np.all(stock >= 0)
    assert np.all(stock <= 10000)
    assert stock[-1] == pytest.approx(7500, abs=0.1)


def test_fish_stock_rejects_invalid_harvest_rate():
    """Harvest rates outside 0 to 1 should be rejected."""
    with pytest.raises(ValueError):
        FishStock(harvest_rate=1.5)


def test_price_model_is_reproducible_and_bounded():
    """Identical seeds should give identical bounded price paths."""
    prices_a = PriceModel(seed=42).simulate(weeks=52)
    prices_b = PriceModel(seed=42).simulate(weeks=52)

    assert len(prices_a) == 53
    assert np.array_equal(prices_a, prices_b)
    assert np.all(prices_a >= 9000)
    assert np.all(prices_a <= 16000)
    assert prices_a[0] == 12000


def test_price_model_rejects_invalid_price_bounds():
    """Invalid price limits should raise a ValueError."""
    with pytest.raises(ValueError):
        PriceModel(initial_price=17000)


def test_risk_assessor_classifies_cv():
    """CV thresholds should produce the expected risk categories."""
    assessor = RiskAssessor()

    assert assessor.classify(0.05) == "Low"
    assert assessor.classify(0.10) == "Moderate"
    assert assessor.classify(0.249) == "Moderate"
    assert assessor.classify(0.25) == "High"


def test_risk_assessor_calculates_cv():
    """CV should equal sample standard deviation divided by mean."""
    assessor = RiskAssessor()
    revenues = [100, 110, 90, 100]

    expected_cv = np.std(revenues, ddof=1) / np.mean(revenues)

    assert assessor.coefficient_of_variation(revenues) == pytest.approx(
        expected_cv
    )


def test_risk_assessor_rejects_insufficient_revenue_data():
    """CV calculation requires at least two revenue values."""
    assessor = RiskAssessor()

    with pytest.raises(ValueError):
        assessor.coefficient_of_variation([100])
