from pathlib import Path

import numpy as np
import pytest

from dask.distributed import Client


@pytest.fixture(scope="session", name="dask_client")
def dask_client():
    """Create a single client for use by all unit test cases."""
    client = Client(n_workers=3, threads_per_worker=1)
    yield client
    client.close()


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent.parent / "data"


@pytest.fixture
def data_catalog_dir(test_data_dir):
    return test_data_dir / "DATA"


@pytest.fixture
def rand_catalog_dir(test_data_dir):
    return test_data_dir / "RAND"


@pytest.fixture
def raw_catalog_dir(test_data_dir):
    return test_data_dir / "RAW"


@pytest.fixture
def dr7_lrg_catalog_dir(test_data_dir):
    return test_data_dir / "DR7-lrg"


@pytest.fixture
def dr7_lrg_rand_catalog_dir(test_data_dir):
    return test_data_dir / "DR7-lrg-rand"


@pytest.fixture
def w_acf_nat(test_data_dir):
    return np.load(test_data_dir / "correlations" / "w_acf_nat.npy")


@pytest.fixture
def w_acf_nat_true(test_data_dir):
    return np.load(test_data_dir / "correlations" / "w_acf_nat_true.npy")
