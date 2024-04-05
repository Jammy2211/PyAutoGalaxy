from __future__ import division, print_function
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():
    lp = ag.lp.Sky(
        intensity=3.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0], [3.0, 0.0]]))

    assert image[0] == pytest.approx(3.0, 1e-3)
    assert image[1] == pytest.approx(3.0, 1e-3)
