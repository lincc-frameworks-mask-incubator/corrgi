import numpy as np
import numpy.testing as npt
import pytest
from gundam import gundam
import hipscat

from corrgi.correlation.angular_correlation import AngularCorrelation
from corrgi.corrgi import compute_autocorrelation
from corrgi.dask import compute_autocorrelation_counts
from corrgi.estimators import calculate_tpcf


def test_acf_bins_are_correct(acf_bins_left_edges, acf_bins_right_edges, acf_params):
    bins, _ = gundam.makebins(
        acf_params.nsept,
        acf_params.septmin,
        acf_params.dsept,
        acf_params.logsept,
    )
    all_bins = np.append(acf_bins_left_edges, acf_bins_right_edges[-1])
    assert np.array_equal(bins, all_bins)


def test_acf_counts_are_correct(
    dask_client,
    data_catalog,
    rand_catalog,
    acf_dd_counts,
    acf_rr_counts,
    acf_params,
):
    ang_corr = AngularCorrelation(params=acf_params)
    counts_dd, counts_rr = compute_autocorrelation_counts(
        data_catalog, rand_catalog, ang_corr
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
    estimate = calculate_tpcf(acf_dd_counts, acf_rr_counts, num_galaxies, num_random)
    npt.assert_allclose(acf_nat_estimate, estimate, rtol=2e-3)


def test_acf_e2e(dask_client, data_catalog, rand_catalog, acf_nat_estimate, acf_params):
    estimate = compute_autocorrelation(
        data_catalog, rand_catalog, AngularCorrelation, params=acf_params
    )
    npt.assert_allclose(estimate, acf_nat_estimate, rtol=1e-7)


def test_acf_counts_with_weights_are_correct(
    dask_client,
    acf_gals_weight_catalog,
    acf_rans_weight_catalog,
    acf_dd_counts_with_weights,
    acf_rr_counts_with_weights,
    acf_params,
):
    ang_corr = AngularCorrelation(params=acf_params, use_weights=True)
    counts_dd, counts_rr = compute_autocorrelation_counts(
        acf_gals_weight_catalog, acf_rans_weight_catalog, ang_corr
    )
    npt.assert_allclose(counts_dd, acf_dd_counts_with_weights, rtol=1e-3)
    npt.assert_allclose(counts_rr, acf_rr_counts_with_weights, rtol=2e-3)


def test_acf_weights_not_provided(data_catalog, rand_catalog, acf_params):
    with pytest.raises(ValueError, match="does not exist"):
        compute_autocorrelation(
            data_catalog,
            rand_catalog,
            AngularCorrelation,
            params=acf_params,
            use_weights=True,
        )
