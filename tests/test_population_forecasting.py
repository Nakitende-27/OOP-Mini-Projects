
import numpy as np
import pytest

from src.population_forecasting import DistrictPopulation


def test_valid_district():
    district = DistrictPopulation(
        "Kampala",
        [2015, 2016, 2017],
        [1200, 1250, 1300]
    )

    assert len(district) == 3
    assert district.DistrictName == "Kampala"
    assert district.populations.tolist() == [1200, 1250, 1300]


def test_rejects_different_lengths():
    with pytest.raises(ValueError):
        DistrictPopulation(
            "Kampala",
            [2015, 2016],
            [1200]
        )


def test_rejects_negative_population():
    with pytest.raises(ValueError):
        DistrictPopulation(
            "Kampala",
            [2015, 2016],
            [1200, -50]
        )


def test_rejects_duplicate_years():
    with pytest.raises(ValueError):
        DistrictPopulation(
            "Kampala",
            [2015, 2015],
            [1200, 1250]
        )
