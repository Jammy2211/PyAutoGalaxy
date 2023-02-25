from __future__ import division, print_function
import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():

    shapelet = ag.lp_shapelets.ShapeletCartesian(
        n_y=2, n_x=3, centre=(0.0, 0.0), beta=1.0
    )

    image = shapelet.image_2d_from(grid=np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.1397, 0.0708009]), 1e-4)

    shapelet = ag.lp_shapelets.ShapeletCartesian(
        n_y=2, n_x=3, centre=(0.2, 0.4), beta=1.0
    )

    image = shapelet.image_2d_from(grid=np.array([[0.0, 1.0], [0.5, 0.25]]))

    assert image == pytest.approx(np.array([0.23733, -0.07913]), 1e-4)
