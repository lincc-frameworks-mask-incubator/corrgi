from copy import deepcopy

from munch import Munch


def create_gundam_params(params: Munch) -> tuple[Munch, Munch]:
    """Generates the parameters for the galaxy and random samples.

    Args:
        params (dict): The dictionary with the base parameters.

    Returns:
        A tuple containing the Munch dictionaries of parameters for
        the galaxy samples and the random samples.
    """
    # Disable grid and fill some mock parameters
    params.grid = 0
    params.sbound = [1, 2, 1, 2]
    params.mxh1 = 2
    params.mxh2 = 2
    params.sk1 = [[1, 2], [1, 2]]
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
    par_dr = deepcopy(params)
    par_dr.kind = "thC"
    par_dr.cntid = "DR"
    par_dr.logf = "DR_log"
    par_dr.wfib = False  # don't do fiber corrections in crounts counts
    par_dr.doboot = False  # don't do bootstraping in cross counts
    return par_dd, par_rr
