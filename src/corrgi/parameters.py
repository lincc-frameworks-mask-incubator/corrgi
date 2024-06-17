from copy import deepcopy

from gundam import gundam
from munch import Munch


def generate_gundam_params() -> tuple[Munch, Munch]:
    """Generates the parameters for the galaxy and random samples.

    Returns:
        A tuple containing the Munch dictionaries of parameters for
        the galaxy samples and the random samples.
    """
    par = gundam.packpars(kind="acf", write=False)
    par_dd = deepcopy(par)
    par_dd.kind = "thA"
    par_dd.cntid = "DD"
    par_rr = deepcopy(par)
    par_rr.kind = "thA"
    par_rr.cntid = "RR"
    par_rr.wfib = False  # don't do fiber corrections in random counts
    par_rr.doboot = False  # don't do bootstraping in random counts
    par_dr = deepcopy(par)
    par_dr.kind = "thC"
    par_dr.cntid = "DR"
    par_dr.wfib = False  # don't do fiber corrections in crounts counts
    par_dr.doboot = False  # don't do bootstraping in cross counts
    return par_dd, par_rr
