import numpy as np
import pytest

import autogalaxy as ag

from autogalaxy.profiles.light_profiles.light_profiles_linear import LightProfileLinear
from autogalaxy.profiles.light_profiles.light_profiles_linear import (
    LightProfileLinearObjFuncList,
)


def test__mapping_matrix_from(sub_grid_2d_7x7, blurring_grid_2d_7x7, convolver_7x7):

    lp_0 = ag.lp.EllSersic(intensity=1.0)
    lp_1 = ag.lp.EllSersic(intensity=2.0)

    lp_linear_obj_func_list = LightProfileLinearObjFuncList(
        grid=sub_grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        convolver=convolver_7x7,
        light_profile_list=[lp_0, lp_1],
    )

    assert lp_linear_obj_func_list.pixels == 2

    lp_0_image = lp_0.image_2d_from(grid=sub_grid_2d_7x7)
    lp_1_image = lp_1.image_2d_from(grid=sub_grid_2d_7x7)

    assert lp_linear_obj_func_list.mapping_matrix[:, 0] == pytest.approx(
        lp_0_image.binned, 1.0e-4
    )
    assert lp_linear_obj_func_list.mapping_matrix[:, 1] == pytest.approx(
        lp_1_image.binned, 1.0e-4
    )

    lp_0_blurred_image = lp_0.blurred_image_2d_from(
        grid=sub_grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        convolver=convolver_7x7,
    )

    lp_1_blurred_image = lp_1.blurred_image_2d_from(
        grid=sub_grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        convolver=convolver_7x7,
    )

    assert lp_linear_obj_func_list.operated_mapping_matrix_override[
        :, 0
    ] == pytest.approx(lp_0_blurred_image, 1.0e-4)
    assert lp_linear_obj_func_list.operated_mapping_matrix_override[
        :, 1
    ] == pytest.approx(lp_1_blurred_image, 1.0e-4)


def test__lp_from():

    lp_linear = ag.lp_linear.EllSersic(centre=(1.0, 2.0))

    lp_non_linear = lp_linear.lp_instance_from(intensity=3.0)

    assert not isinstance(lp_non_linear, LightProfileLinear)
    assert type(lp_non_linear) is ag.lp.EllSersic
    assert lp_non_linear.centre == (1.0, 2.0)
    assert lp_non_linear.intensity == 3.0
