import numpy as np
import pytest

import autogalaxy as ag

from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.profiles.light.linear import (
    LightProfileLinearObjFuncList,
)


def test__mapping_matrix_from(grid_2d_7x7, blurring_grid_2d_7x7, convolver_7x7):
    lp_0 = ag.lp_linear.Sersic(effective_radius=1.0)
    lp_1 = ag.lp_linear.Sersic(effective_radius=2.0)

    lp_linear_obj_func_list = LightProfileLinearObjFuncList(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        convolver=convolver_7x7,
        light_profile_list=[lp_0, lp_1],
    )

    assert lp_linear_obj_func_list.params == 2

    lp_0_image = lp_0.image_2d_from(grid=grid_2d_7x7)
    lp_1_image = lp_1.image_2d_from(grid=grid_2d_7x7)

    assert lp_linear_obj_func_list.mapping_matrix[:, 0] == pytest.approx(
        lp_0_image, 1.0e-4
    )
    assert lp_linear_obj_func_list.mapping_matrix[:, 1] == pytest.approx(
        lp_1_image, 1.0e-4
    )

    lp_0_blurred_image = lp_0.blurred_image_2d_from(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        convolver=convolver_7x7,
    )

    lp_1_blurred_image = lp_1.blurred_image_2d_from(
        grid=grid_2d_7x7,
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
    lp_linear = ag.lp_linear.Sersic(centre=(1.0, 2.0))

    lp_non_linear = lp_linear.lp_instance_from(
        linear_light_profile_intensity_dict={lp_linear: 3.0}
    )

    assert not isinstance(lp_non_linear, LightProfileLinear)
    assert type(lp_non_linear) is ag.lp.Sersic
    assert lp_non_linear.centre == (1.0, 2.0)
    assert lp_non_linear.intensity == 3.0
