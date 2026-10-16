import numpy as np
import pytest

from src.rainfall import Region, CropRule


def test_region_summary():
    rainfall = np.array([10, 20, 30, 40, 50, 60,
                         70, 80, 90, 100, 110, 120])
    region = Region("Kampala", rainfall)

    assert region.annual_total() == 780.0
    assert region.mean_rainfall() == 65.0
    assert region.wettest_month() == "Dec"
    assert region.driest_month() == "Jan"
    assert region.coefficient_of_variation() == pytest.approx(
        np.std(rainfall) / np.mean(rainfall)
    )


def test_region_rejects_wrong_number_of_months():
    with pytest.raises(ValueError):
        Region("Kampala", np.array([10, 20, 30]))


def test_region_rejects_negative_rainfall():
    rainfall = np.array([10] * 11 + [-5])

    with pytest.raises(ValueError):
        Region("Kampala", rainfall)


def test_region_rejects_non_finite_rainfall():
    rainfall = np.array([10.0] * 11 + [np.nan])

    with pytest.raises(ValueError):
        Region("Kampala", rainfall)


def test_crop_rule_classification():
    maize = CropRule("maize", 50, 150)

    assert maize.classify(30) == "Drought risk"
    assert maize.classify(100) == "Good for maize"
    assert maize.classify(160) == "Waterlogging risk"


def test_crop_rule_includes_threshold_boundaries():
    maize = CropRule("maize", 50, 150)

    assert maize.classify(50) == "Good for maize"
    assert maize.classify(150) == "Good for maize"


def test_crop_rule_rejects_invalid_thresholds():
    with pytest.raises(ValueError):
        CropRule("maize", 150, 50)


def test_crop_rule_rejects_negative_rainfall():
    maize = CropRule("maize", 50, 150)

    with pytest.raises(ValueError):
        maize.classify(-10)
