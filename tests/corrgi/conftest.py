from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def data_catalog_dir(test_data_dir):
    return test_data_dir / "DATA"


@pytest.fixture
def raw_catalog_dir(test_data_dir):
    return test_data_dir / "RAW"


@pytest.fixture
def dr7_lrg_catalog_dir(test_data_dir):
    return test_data_dir / "DR7-lrg"


@pytest.fixture
def dr7_lrg_rand_catalog_dir(test_data_dir):
    return test_data_dir / "DR7-lrg-rand"
