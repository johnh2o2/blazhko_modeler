from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import zip
from builtins import range
from builtins import object
import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..blazhko import BlazhkoPeriodogram, BlazhkoModel

seed = 100

rand = np.random.RandomState(seed)


@pytest.fixture
def data(sigma=0.1, ndata=500, freq=3., snr=1000, t0=0.):

    t = np.sort(rand.rand(ndata)) + t0
    y = snr * sigma * np.cos(2 * np.pi * freq * t) / np.sqrt(len(t))

    y += sigma * rand.randn(len(t))

    err = sigma * np.ones_like(y)

    return t, y, err


@pytest.mark.parametrize('ndata', [1, 10, 100])
@pytest.mark.parametrize('freq', [0.1, 10., 100.])
def test_something(data, ndata, freq):
    pass
