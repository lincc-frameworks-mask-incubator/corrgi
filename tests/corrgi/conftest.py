from pathlib import Path

import gundam
import numpy as np
import pandas as pd
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
def hipscat_catalogs_dir(test_data_dir):
    return test_data_dir / "hipscat"


@pytest.fixture
def acf_expected_results(test_data_dir):
    return test_data_dir / "expected_results" / "acf"


@pytest.fixture
def data_catalog_dir(hipscat_catalogs_dir):
    return hipscat_catalogs_dir / "DATA"


@pytest.fixture
def dr7_lrg_catalog_dir(hipscat_catalogs_dir):
    return hipscat_catalogs_dir / "DR7-lrg"


@pytest.fixture
def dr7_lrg_rand_catalog_dir(hipscat_catalogs_dir):
    return hipscat_catalogs_dir / "DR7-lrg-rand"


@pytest.fixture
def rand_catalog_dir(hipscat_catalogs_dir):
    return hipscat_catalogs_dir / "RAND"


@pytest.fixture
def acf_bins_left_edges(acf_expected_results):
    return np.load(acf_expected_results / "l_binedges_acf.npy")


@pytest.fixture
def acf_bins_right_edges(acf_expected_results):
    return np.load(acf_expected_results / "r_binedges_acf.npy")


@pytest.fixture
def acf_dd_counts(acf_expected_results):
    return np.load(acf_expected_results / "dd_acf.npy")


@pytest.fixture
def acf_rr_counts(acf_expected_results):
    return np.load(acf_expected_results / "rr_acf.npy")


@pytest.fixture
def acf_nat_estimate(acf_expected_results):
    return np.load(acf_expected_results / "w_acf_nat.npy")


@pytest.fixture
def single_data_partition(data_catalog_dir):
    return pd.read_parquet(data_catalog_dir / "Norder=0" / "Dir=0" / "Npix=1.parquet")


@pytest.fixture
def corr_bins():
    bins, _ = gundam.makebins(33, 0.01, 0.1, 1)
    return bins


@pytest.fixture
def autocorr_params():
    params = gundam.packpars(kind="acf", write=False)

    params.dsept = 0.1
    params.nsept = 33
    params.septmin = 0.01

    params.kind = "thA"
    params.cntid = "DD"
    params.logf = "DD_log"

    # Disable grid and fill some mock parameters
    params.grid = 0
    params.sbound = [1, 2, 1, 2]
    params.mxh1 = 2
    params.mxh2 = 2
    params.sk1 = [[1, 2], [1, 2]]
    return params
