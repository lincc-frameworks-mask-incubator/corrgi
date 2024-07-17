import pytest
from corrgi.correlation.projected_correlation import ProjectedCorrelation
from corrgi.corrgi import compute_autocorrelation
import numpy.testing as npt

from corrgi.estimators.natural_estimator import NaturalEstimator


def test_pcf_natural_counts_are_correct(
    dask_client, data_catalog, rand_catalog, pcf_dd_counts, pcf_rr_counts, pcf_params
):
    estimator = NaturalEstimator(ProjectedCorrelation(params=pcf_params))
    counts_dd, counts_rr, _ = estimator.compute_autocorrelation_counts(
        data_catalog, rand_catalog
    )
    npt.assert_allclose(counts_dd, pcf_dd_counts, rtol=1e-3)
    npt.assert_allclose(counts_rr, pcf_rr_counts, rtol=2e-3)


def test_pcf_natural_estimate_is_correct(
    dask_client, data_catalog, rand_catalog, pcf_nat_estimate, pcf_params
):
    pcf_params.estimator = "NAT"
    estimate = compute_autocorrelation(
        data_catalog, rand_catalog, ProjectedCorrelation, params=pcf_params
    )
    npt.assert_allclose(estimate, pcf_nat_estimate, rtol=1e-3)


def test_pcf_counts_with_weights_are_correct(
    dask_client,
    pcf_gals_weight_catalog,
    pcf_rans_weight_catalog,
    pcf_dd_counts_with_weights,
    pcf_rr_counts_with_weights,
    pcf_params,
):
    estimator = NaturalEstimator(
        ProjectedCorrelation(params=pcf_params, use_weights=True)
    )
    counts_dd, counts_rr, _ = estimator.compute_autocorrelation_counts(
        pcf_gals_weight_catalog, pcf_rans_weight_catalog
    )
    npt.assert_allclose(counts_dd, pcf_dd_counts_with_weights, rtol=1e-3)
    npt.assert_allclose(counts_rr, pcf_rr_counts_with_weights, rtol=2e-3)


def test_pcf_catalog_has_no_redshift(data_catalog, rand_catalog, pcf_params):
    with pytest.raises(ValueError, match="ph_z not found"):
        compute_autocorrelation(
            data_catalog,
            rand_catalog,
            ProjectedCorrelation,
            params=pcf_params,
            redshift_column="ph_z",
        )