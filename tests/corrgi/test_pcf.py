import pytest
from corrgi.correlation.projected_correlation import ProjectedCorrelation
from corrgi.corrgi import compute_autocorrelation
from corrgi.dask import compute_autocorrelation_counts
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
