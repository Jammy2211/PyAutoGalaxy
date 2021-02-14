from __future__ import division, print_function

import math

import numpy as np
import pytest
import scipy.special

import autogalaxy as ag
from autogalaxy.mock import mock

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestPointSourceFlux:
    def test__constructor(self):

        point_source = ag.ps.PointSourceFlux(centre=(0.0, 0.0), flux=0.1)

        assert point_source.centre == (0.0, 0.0)
        assert point_source.flux == 0.1
