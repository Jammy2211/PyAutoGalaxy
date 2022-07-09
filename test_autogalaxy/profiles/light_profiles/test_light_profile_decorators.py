import numpy as np
import pytest
from typing import Optional

import autoarray as aa
import autogalaxy as ag

from autogalaxy.profiles.light_profiles.light_profile_decorators import (
    check_operated_only,
)


class MockLightProfile(ag.lp.LightProfile):
    @check_operated_only
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ):
        return np.ones(shape=(3, 3))


class MockLightProfileOperated(ag.lp_operated.LightProfileOperated):
    @check_operated_only
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ):
        return np.ones(shape=(3, 3))


def test__decorator_changes_behaviour_correctly():

    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    lp = ag.lp.EllGaussian()

    lp_image_2d = lp.image_2d_from(grid=grid)

    image_2d = lp.image_2d_from(grid=grid)
    assert image_2d == pytest.approx(lp_image_2d, 1.0e-4)

    image_2d = lp.image_2d_from(grid=grid, operated_only=True)
    assert image_2d == pytest.approx(np.zeros(shape=(9,)), 1.0e-4)

    image_2d = lp.image_2d_from(grid=grid, operated_only=False)
    assert image_2d == pytest.approx(lp_image_2d, 1.0e-4)

    lp = ag.lp_operated.EllGaussian()

    image_2d = lp.image_2d_from(grid=grid)
    assert image_2d == pytest.approx(lp_image_2d, 1.0e-4)

    image_2d = lp.image_2d_from(grid=grid, operated_only=True)
    assert image_2d == pytest.approx(lp_image_2d, 1.0e-4)

    image_2d = lp.image_2d_from(grid=grid, operated_only=False)
    assert image_2d == pytest.approx(np.zeros(shape=(9,)), 1.0e-4)
