import numpy as np
import pytest

import autogalaxy as ag

from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.profiles.light.linear import (
    LightProfileLinearObjFuncList,
)


def test__params__two_light_profiles__equals_two(
    grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3
):
    lp_0 = ag.lp_linear.Sersic(effective_radius=1.0)
    lp_1 = ag.lp_linear.Sersic(effective_radius=2.0)

    lp_linear_obj_func_list = LightProfileLinearObjFuncList(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        psf=psf_3x3,
        light_profile_list=[lp_0, lp_1],
    )

    assert lp_linear_obj_func_list.params == 2


def test__mapping_matrix__columns_match_individual_profile_images(
    grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3
):
    lp_0 = ag.lp_linear.Sersic(effective_radius=1.0)
    lp_1 = ag.lp_linear.Sersic(effective_radius=2.0)

    lp_linear_obj_func_list = LightProfileLinearObjFuncList(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        psf=psf_3x3,
        light_profile_list=[lp_0, lp_1],
    )

    lp_0_image = lp_0.image_2d_from(grid=grid_2d_7x7)
    lp_1_image = lp_1.image_2d_from(grid=grid_2d_7x7)

    assert lp_linear_obj_func_list.mapping_matrix[:, 0] == pytest.approx(
        lp_0_image.array, 1.0e-4
    )
    assert lp_linear_obj_func_list.mapping_matrix[:, 1] == pytest.approx(
        lp_1_image.array, 1.0e-4
    )


def test__operated_mapping_matrix__columns_match_individual_blurred_images(
    grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3
):
    lp_0 = ag.lp_linear.Sersic(effective_radius=1.0)
    lp_1 = ag.lp_linear.Sersic(effective_radius=2.0)

    lp_linear_obj_func_list = LightProfileLinearObjFuncList(
        grid=grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
        psf=psf_3x3,
        light_profile_list=[lp_0, lp_1],
    )

    lp_0_blurred_image = lp_0.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    lp_1_blurred_image = lp_1.blurred_image_2d_from(
        grid=grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    assert lp_linear_obj_func_list.operated_mapping_matrix_override[
        :, 0
    ] == pytest.approx(lp_0_blurred_image.array, 1.0e-4)
    assert lp_linear_obj_func_list.operated_mapping_matrix_override[
        :, 1
    ] == pytest.approx(lp_1_blurred_image.array, 1.0e-4)


def test__lp_instance_from__returns_non_linear_instance_with_correct_type_and_centre():
    lp_linear = ag.lp_linear.Sersic(centre=(1.0, 2.0))

    lp_non_linear = lp_linear.lp_instance_from(
        linear_light_profile_intensity_dict={lp_linear: 3.0}
    )

    assert not isinstance(lp_non_linear, LightProfileLinear)
    assert type(lp_non_linear) is ag.lp.Sersic
    assert lp_non_linear.centre == (1.0, 2.0)


def test__lp_instance_from__returns_instance_with_correct_intensity():
    lp_linear = ag.lp_linear.Sersic(centre=(1.0, 2.0))

    lp_non_linear = lp_linear.lp_instance_from(
        linear_light_profile_intensity_dict={lp_linear: 3.0}
    )

    assert lp_non_linear.intensity == 3.0


def test__pytree_token_is_int_and_unique():
    lp_0 = ag.lp_linear.Sersic()
    lp_1 = ag.lp_linear.Sersic()

    assert isinstance(lp_0.pytree_token, int)
    assert isinstance(lp_1.pytree_token, int)
    assert lp_0.pytree_token != lp_1.pytree_token

    assert isinstance(hash(lp_0), int)
    assert hash(lp_0) == hash(lp_0)
    assert hash(lp_0) != hash(lp_1)


def test__getstate__omits_pytree_token():
    lp = ag.lp_linear.Sersic()
    state = lp.__getstate__()

    assert "pytree_token" not in state
    assert "effective_radius" in state


def test__setstate__assigns_fresh_pytree_token_when_missing():
    lp = ag.lp_linear.Sersic()
    state = lp.__getstate__()

    restored = ag.lp_linear.Sersic.__new__(ag.lp_linear.Sersic)
    restored.__setstate__(state)

    assert isinstance(restored.pytree_token, int)
    assert isinstance(hash(restored), int)


def test__pickle_roundtrip_preserves_int_hash():
    import pickle

    lp = ag.lp_linear.Sersic()
    restored = pickle.loads(pickle.dumps(lp))

    assert isinstance(hash(restored), int)
    assert isinstance(restored.pytree_token, int)
    assert restored.effective_radius == lp.effective_radius


def test__setstate__preserves_pytree_token_when_present():
    lp = ag.lp_linear.Sersic()
    state_with_token = dict(lp.__dict__)

    restored = ag.lp_linear.Sersic.__new__(ag.lp_linear.Sersic)
    restored.__setstate__(state_with_token)

    assert restored.pytree_token == lp.pytree_token
