import numpy as np
import pytest

import autogalaxy as ag


def test__image_2d_from__does_not_include_linear_light_profiles(grid_2d_7x7):
    lp = ag.lp.Sersic(intensity=0.1)

    lp_linear = ag.lp_linear.Sersic(effective_radius=2.0, sersic_index=2.0)

    lp_image = lp.image_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(light_profile_list=[lp, lp_linear])

    image = basis.image_2d_from(grid=grid_2d_7x7)

    assert (image == lp_image).all()


def test__image_2d_from__operated_only_input(grid_2d_7x7, lp_0, lp_operated_0):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(light_profile_list=[lp_0, lp_operated_0])

    image_2d = basis.image_2d_from(grid=grid_2d_7x7, operated_only=False)
    assert (image_2d == image_2d_not_operated).all()

    image_2d = basis.image_2d_from(grid=grid_2d_7x7, operated_only=True)
    assert (image_2d == image_2d_operated).all()

    image_2d = basis.image_2d_from(grid=grid_2d_7x7, operated_only=None)
    assert (image_2d == image_2d_not_operated + image_2d_operated).all()


def test__image_2d_list_from__operated_only_input(grid_2d_7x7, lp_0, lp_operated_0):
    image_2d_not_operated = lp_0.image_2d_from(grid=grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=grid_2d_7x7)

    basis = ag.lp_basis.Basis(light_profile_list=[lp_0, lp_operated_0])

    image_2d_list = basis.image_2d_list_from(grid=grid_2d_7x7, operated_only=False)
    assert (image_2d_list[0] == image_2d_not_operated).all()
    assert (image_2d_list[1] == np.zeros((9))).all()

    image_2d_list = basis.image_2d_list_from(grid=grid_2d_7x7, operated_only=True)
    assert (image_2d_list[0] == np.zeros((9))).all()
    assert (image_2d_list[1] == image_2d_operated).all()

    image_2d_list = basis.image_2d_list_from(grid=grid_2d_7x7, operated_only=None)
    assert (
        image_2d_list[0] + image_2d_list[1] == image_2d_not_operated + image_2d_operated
    ).all()
