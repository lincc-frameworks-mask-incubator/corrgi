import numpy as np

from corrgi.parameters import create_gundam_params


def test_create_gundam_params():
    params = create_gundam_params(kind="acf", dsept=0.10, nsept=33, septmin=0.01)
    assert params.kind == "acf"
    assert params.grid == 0
    assert np.array_equal(params.sbound, [1, 2, 1, 2])
    assert params.mxh1 == 2
    assert params.mxh2 == 2
    assert np.array_equal(params.sk1, [[1, 2], [1, 2]])
    assert params.dsept == 0.10
    assert params.nsept == 33
    assert params.septmin == 0.01
