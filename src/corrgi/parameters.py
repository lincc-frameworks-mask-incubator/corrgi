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
    params.sbound = 0  # TODO: check what this is
    par_dd = deepcopy(params)
    par_dd.kind = "thA"
    par_dd.cntid = "DD"
    par_rr = deepcopy(params)
    par_rr.kind = "thA"
    par_rr.cntid = "RR"
    par_rr.wfib = False  # don't do fiber corrections in random counts
    par_rr.doboot = False  # don't do bootstraping in random counts
    par_dr = deepcopy(params)
    par_dr.kind = "thC"
    par_dr.cntid = "DR"
    par_dr.wfib = False  # don't do fiber corrections in crounts counts
    par_dr.doboot = False  # don't do bootstraping in cross counts
    return par_dd, par_rr
