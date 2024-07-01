import hipscat
import lsdb
import numpy as np
import numpy.testing as npt
from gundam import gundam

from corrgi.correlation.angular_correlation import AngularCorrelation
from corrgi.corrgi import compute_autocorrelation
from corrgi.dask import compute_autocorrelation_counts
from corrgi.estimators import calculate_natural_estimate


def test_acf_bins_are_correct(
    acf_bins_left_edges, acf_bins_right_edges, autocorr_params
):
    bins, _ = gundam.makebins(
        autocorr_params.nsept,
        autocorr_params.septmin,
        autocorr_params.dsept,
        autocorr_params.logsept,
    )
    all_bins = np.append(acf_bins_left_edges, acf_bins_right_edges[-1])
    assert np.array_equal(bins, all_bins)


def test_acf_counts_are_correct(
    dask_client,
    data_catalog_dir,
    rand_catalog_dir,
    acf_dd_counts,
    acf_rr_counts,
    autocorr_params,
):
    galaxy_catalog = lsdb.read_hipscat(data_catalog_dir)
    random_catalog = lsdb.read_hipscat(rand_catalog_dir)
    assert isinstance(galaxy_catalog, lsdb.Catalog)
    assert isinstance(random_catalog, lsdb.Catalog)
    counts_dd, counts_rr = compute_autocorrelation_counts(
        AngularCorrelation, galaxy_catalog, random_catalog, autocorr_params
    )
    npt.assert_allclose(counts_dd, acf_dd_counts, rtol=1e-3)
    npt.assert_allclose(counts_rr, acf_rr_counts, rtol=2e-3)


def test_acf_natural_estimate_is_correct(
    data_catalog_dir,
    rand_catalog_dir,
    acf_dd_counts,
    acf_rr_counts,
    acf_nat_estimate,
):
    galaxy_hc_catalog = hipscat.read_from_hipscat(data_catalog_dir)
    random_hc_catalog = hipscat.read_from_hipscat(rand_catalog_dir)
    num_galaxies = galaxy_hc_catalog.catalog_info.total_rows
    num_random = random_hc_catalog.catalog_info.total_rows
    estimate = calculate_natural_estimate(
        acf_dd_counts, acf_rr_counts, num_galaxies, num_random
    )
    npt.assert_allclose(acf_nat_estimate, estimate, rtol=2e-3)


def test_acf_e2e(
    dask_client, data_catalog_dir, rand_catalog_dir, acf_nat_estimate, autocorr_params
):
    galaxy_catalog = lsdb.read_hipscat(data_catalog_dir)
    random_catalog = lsdb.read_hipscat(rand_catalog_dir)
    assert isinstance(galaxy_catalog, lsdb.Catalog)
    assert isinstance(random_catalog, lsdb.Catalog)
    estimate = compute_autocorrelation(
        AngularCorrelation, galaxy_catalog, random_catalog, autocorr_params
    )
    npt.assert_allclose(estimate, acf_nat_estimate, rtol=1e-7)
