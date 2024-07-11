import numpy as np
import numpy.testing as npt
import pytest

from corrgi.correlation.angular_correlation import AngularCorrelation
from corrgi.corrgi import compute_autocorrelation
from corrgi.estimators.natural_estimator import NaturalEstimator


def test_acf_bins_are_correct(acf_bins_left_edges, acf_bins_right_edges, acf_params):
    bins = AngularCorrelation(params=acf_params).make_bins()
    all_bins = np.append(acf_bins_left_edges, acf_bins_right_edges[-1])
    assert np.array_equal(bins, all_bins)


def test_acf_natural_counts_are_correct(
    dask_client, data_catalog, rand_catalog, acf_dd_counts, acf_rr_counts, acf_params
):
    estimator = NaturalEstimator(AngularCorrelation(params=acf_params))
    counts_dd, counts_rr = estimator.compute_autocorrelation_counts(
        data_catalog, rand_catalog
    )
    npt.assert_allclose(counts_dd, acf_dd_counts, rtol=1e-3)
    npt.assert_allclose(counts_rr, acf_rr_counts, rtol=2e-3)


def test_acf_natural_estimate_is_correct(
    dask_client, data_catalog, rand_catalog, acf_nat_estimate, acf_params
):
    acf_params.estimator = "NAT"
    estimate = compute_autocorrelation(
        data_catalog, rand_catalog, AngularCorrelation, params=acf_params
    )
    npt.assert_allclose(estimate, acf_nat_estimate, rtol=1e-7)


def test_acf_natural_counts_with_weights_are_correct(
    dask_client,
    acf_gals_weight_catalog,
    acf_rans_weight_catalog,
    acf_dd_counts_with_weights,
    acf_rr_counts_with_weights,
    acf_params,
):
    estimator = NaturalEstimator(
        AngularCorrelation(params=acf_params, use_weights=True)
    )
    counts_dd, counts_rr = estimator.compute_autocorrelation_counts(
        acf_gals_weight_catalog, acf_rans_weight_catalog
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
