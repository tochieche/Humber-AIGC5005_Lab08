"""Small, repeatable demo data for the checks."""

import pytest

from bikes_data import make_rentals
from prepare import prepare_data


@pytest.fixture(scope="module")
def sample():
    return prepare_data(make_rentals(n=300), seed=42)
