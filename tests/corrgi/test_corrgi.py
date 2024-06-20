import hipscat
import lsdb
import numpy as np
import numpy.testing as npt

from corrgi.corrgi import compute_autocorrelation
from gundam import gundam

from corrgi.dask import compute_autocorrelation_counts
from corrgi.estimators import calculate_natural_estimate
from corrgi.parameters import create_gundam_params


def test_acf1_bins_are_correct(acf1_bins_left_edges, acf1_bins_right_edges):
    params = create_gundam_params(kind="acf1", dsept=0.10, nsept=33, septmin=0.01)
    bins, _ = gundam.makebins(
        params.nsept, params.septmin, params.dsept, params.logsept
    )
    all_bins = np.append(acf1_bins_left_edges, acf1_bins_right_edges[-1])
    assert np.array_equal(all_bins, bins)


def test_acf1_counts_are_correct(
    dask_client, data_catalog_dir, rand_catalog_dir, acf1_dd_counts, acf1_rr_counts
):
    params = create_gundam_params(kind="acf1", dsept=0.10, nsept=33, septmin=0.01)
    galaxy_catalog = lsdb.read_hipscat(data_catalog_dir)
    random_catalog = lsdb.read_hipscat(rand_catalog_dir)
    assert isinstance(galaxy_catalog, lsdb.Catalog)
    assert isinstance(random_catalog, lsdb.Catalog)
    counts_dd, counts_rr = compute_autocorrelation_counts(
        galaxy_catalog, random_catalog, params
    )
    assert np.array_equal(counts_dd, acf1_dd_counts)
    assert np.array_equal(counts_rr, acf1_rr_counts)


def test_acf1_natural_estimate_is_correct(
    data_catalog_dir,
    rand_catalog_dir,
    acf1_dd_counts,
    acf1_rr_counts,
    acf1_nat_estimate,
):
    galaxy_hc_catalog = hipscat.read_from_hipscat(data_catalog_dir)
    random_hc_catalog = hipscat.read_from_hipscat(rand_catalog_dir)
    num_galaxies = galaxy_hc_catalog.catalog_info.total_rows
    num_random = random_hc_catalog.catalog_info.total_rows
    estimate = calculate_natural_estimate(
        acf1_dd_counts, acf1_rr_counts, num_galaxies, num_random
    )
    assert np.array_equal(acf1_nat_estimate, estimate)


def test_acf1_e2e(dask_client, data_catalog_dir, rand_catalog_dir, acf1_nat_estimate):
    params = create_gundam_params(kind="acf1", dsept=0.10, nsept=33, septmin=0.01)
    galaxy_catalog = lsdb.read_hipscat(data_catalog_dir)
    random_catalog = lsdb.read_hipscat(rand_catalog_dir)
    assert isinstance(galaxy_catalog, lsdb.Catalog)
    assert isinstance(random_catalog, lsdb.Catalog)
    estimate = compute_autocorrelation(galaxy_catalog, random_catalog, params)
    npt.assert_allclose(estimate, acf1_nat_estimate, rtol=1e-7)
