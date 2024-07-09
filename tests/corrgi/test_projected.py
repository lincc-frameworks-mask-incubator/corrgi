import pytest
from corrgi.correlation.projected_correlation import ProjectedCorrelation
from corrgi.corrgi import compute_autocorrelation
from corrgi.dask import compute_autocorrelation_counts, compute_crosscorrelation_counts
import numpy.testing as npt


def test_pcf_counts_are_correct(
    dask_client, data_catalog, rand_catalog, pcf_dd_counts, pcf_rr_counts, pcf_params
):
    proj_corr = ProjectedCorrelation(params=pcf_params)
    counts_dd, counts_rr = compute_autocorrelation_counts(
        data_catalog, rand_catalog, proj_corr
    )
    expected_dd, expected_rr = counts_dd.transpose([1, 0]), counts_rr.transpose([1, 0])
    npt.assert_allclose(expected_dd, pcf_dd_counts, rtol=1e-3)
    npt.assert_allclose(expected_rr, pcf_rr_counts, rtol=2e-3)


def test_pcf_catalog_has_no_redshift(data_catalog, rand_catalog, pcf_params):
    with pytest.raises(ValueError, match="ph_z not found"):
        compute_autocorrelation(
            data_catalog,
            rand_catalog,
            ProjectedCorrelation,
            params=pcf_params,
            redshift_column="ph_z",
        )


def test_pccf_with_weights(
    dask_client,
    pcf_gals_weight_catalog,
    pcf_gals1_weight_catalog,
    pcf_rans_weight_catalog,
    pccf_cd_counts,
    pccf_cr_counts,
    pcf_params,
):
    proj_corr = ProjectedCorrelation(params=pcf_params, use_weights=True)
    counts_cd, counts_cr = compute_crosscorrelation_counts(
        pcf_gals_weight_catalog,
        pcf_gals1_weight_catalog,
        pcf_rans_weight_catalog,
        proj_corr,
    )
    expected_cd, expected_cr = counts_cd.transpose([1, 0]), counts_cr.transpose([1, 0])
    npt.assert_allclose(expected_cd, pccf_cd_counts, rtol=1e-3)
    npt.assert_allclose(expected_cr, pccf_cr_counts, rtol=2e-3)
