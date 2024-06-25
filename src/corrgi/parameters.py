from copy import deepcopy

from gundam import gundam
from munch import Munch


def create_gundam_params(kind: str, **kwargs) -> Munch:
    """Generates the Gundam parameters for a specific kind.

    Args:
        kind (str): The type of correlation function (e.g. acf).
        **kwargs: Additional gundam parameters to set/override.

    Returns:
        The dictionary of gundam parameters.
    """
    params = gundam.packpars(kind=kind, write=False)
    # Disable grid and fill its unused parameters
    params.grid = 0
    params.autogrid = False
    params.sbound = [1, 2, 1, 2]
    params.mxh1 = 2
    params.mxh2 = 2
    params.sk1 = [[1, 2], [1, 2]]
    # Append any additional params
    return Munch({**params, **kwargs})


def generate_dd_rr_params(params: Munch) -> tuple[Munch, Munch]:
    """Generate the DD and RR parameters."""
    # Create the parameters for both catalogs
    par_dd = deepcopy(params)
    par_dd.kind = "thC"
    par_dd.cntid = "DD"
    par_dd.logf = "DD_log"
    par_rr = deepcopy(params)
    par_rr.kind = "thC"
    par_rr.cntid = "RR"
    par_rr.logf = "RR_log"
    par_rr.wfib = False  # don't do fiber corrections in random counts
    par_rr.doboot = False  # don't do bootstraping in random counts
    return par_dd, par_rr


def generate_dr_params(params: Munch) -> Munch:
    """Generate the DR parameters used int the cross-correlation."""
    par_dr = deepcopy(params)
    par_dr.kind = "thC"
    par_dr.cntid = "DR"
    par_dr.logf = "DR_log"
    par_dr.wfib = False  # don't do fiber corrections in crounts counts
    par_dr.doboot = False  # don't do bootstraping in cross counts
    return par_dr
