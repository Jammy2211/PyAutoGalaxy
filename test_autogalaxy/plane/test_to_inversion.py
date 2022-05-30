import numpy as np
import pytest

import autogalaxy as ag


def test__lp_linear_func_galaxy_dict_from(lp_0):

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    lp_linear_func_galaxy_dict = plane_to_inversion.lp_linear_func_galaxy_dict_from(
        source_grid_slim=1, source_blurring_grid_slim=2, convolver=3
    )

    assert lp_linear_func_galaxy_dict == {}

    lp_linear_0 = ag.lp_linear.LightProfileLinear()
    lp_linear_1 = ag.lp_linear.LightProfileLinear()
    lp_linear_2 = ag.lp_linear.LightProfileLinear()

    galaxy_0 = ag.Galaxy(
        redshift=0.5, lp_0=lp_0, light_linear_0=lp_linear_0, light_linear_1=lp_linear_1
    )

    galaxy_1 = ag.Galaxy(redshift=0.5, light_linear=lp_linear_2)

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1], redshift=None)

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    lp_linear_func_galaxy_dict = plane_to_inversion.lp_linear_func_galaxy_dict_from(
        source_grid_slim=1, source_blurring_grid_slim=2, convolver=3
    )

    lp_linear_func_list = list(lp_linear_func_galaxy_dict.keys())

    assert lp_linear_func_galaxy_dict[lp_linear_func_list[0]] == galaxy_0
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[1]] == galaxy_0
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[2]] == galaxy_1

    assert lp_linear_func_list[0].light_profile == lp_linear_0
    assert lp_linear_func_list[1].light_profile == lp_linear_1
    assert lp_linear_func_list[2].light_profile == lp_linear_2


def test__sparse_image_plane_grid_list_from(sub_grid_2d_7x7):

    galaxy_pix = ag.Galaxy(
        redshift=0.5,
        pixelization=ag.m.MockPixelization(data_pixelization_grid=[[1.0, 1.0]]),
        regularization=ag.m.MockRegularization(),
    )

    plane = ag.Plane(galaxies=[galaxy_pix], redshift=0.5)

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    sparse_grid = plane_to_inversion.sparse_image_plane_grid_list_from(
        grid=sub_grid_2d_7x7
    )

    assert (sparse_grid == np.array([[1.0, 1.0]])).all()

    # In the ag.m.MockPixelization class the grid is returned if hyper image=None, and grid*hyper image is
    # returned otherwise.

    galaxy_pix = ag.Galaxy(
        redshift=0.5,
        pixelization=ag.m.MockPixelization(
            data_pixelization_grid=np.array([[1.0, 1.0]])
        ),
        regularization=ag.m.MockRegularization(),
        hyper_galaxy_image=2,
    )

    plane = ag.Plane(galaxies=[galaxy_pix], redshift=0.5)

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    sparse_grid = plane_to_inversion.sparse_image_plane_grid_list_from(
        grid=sub_grid_2d_7x7
    )

    assert (sparse_grid == np.array([[2.0, 2.0]])).all()

    # No Galalxies

    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(galaxies=[galaxy_no_pix], redshift=0.5)

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    sparse_grid = plane_to_inversion.sparse_image_plane_grid_list_from(
        grid=sub_grid_2d_7x7
    )

    assert sparse_grid is None


def test__mapper_galaxy_dict_from(sub_grid_2d_7x7):

    galaxy_pix = ag.Galaxy(
        redshift=0.5,
        pixelization=ag.m.MockPixelization(
            mapper=1, data_pixelization_grid=sub_grid_2d_7x7
        ),
        regularization=ag.m.MockRegularization(),
    )
    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(galaxies=[galaxy_no_pix, galaxy_pix], redshift=0.5)

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    mapper_galaxy_dict = plane_to_inversion.mapper_galaxy_dict_from(
        grid=sub_grid_2d_7x7
    )

    mapper_list = list(mapper_galaxy_dict.keys())

    assert mapper_list[0] == 1
    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix

    galaxy_pix_2 = ag.Galaxy(
        redshift=0.5,
        pixelization=ag.m.MockPixelization(
            mapper=2, data_pixelization_grid=sub_grid_2d_7x7
        ),
        regularization=ag.m.MockRegularization(),
    )
    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(
        galaxies=[galaxy_no_pix, galaxy_pix, galaxy_no_pix, galaxy_pix_2], redshift=0.5
    )

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    mapper_galaxy_dict = plane_to_inversion.mapper_galaxy_dict_from(
        grid=sub_grid_2d_7x7
    )

    mapper_list = list(mapper_galaxy_dict.keys())

    assert mapper_list[0] == 1
    assert mapper_list[1] == 2

    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix
    assert mapper_galaxy_dict[mapper_list[1]] == galaxy_pix_2

    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(galaxies=[galaxy_no_pix], redshift=0.5)

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    mapper_galaxy_dict = plane_to_inversion.mapper_galaxy_dict_from(
        grid=sub_grid_2d_7x7
    )

    assert mapper_galaxy_dict == {}


def test__inversion_imaging_from(sub_grid_2d_7x7, masked_imaging_7x7):

    g_linear = ag.Galaxy(redshift=0.5, light_linear=ag.lp_linear.EllSersic())

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g_linear])

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    inversion = plane_to_inversion.inversion_imaging_from(
        dataset=masked_imaging_7x7,
        image=masked_imaging_7x7.image,
        noise_map=masked_imaging_7x7.noise_map,
        w_tilde=masked_imaging_7x7.w_tilde,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert inversion.reconstruction[0] == pytest.approx(0.00543437, 1.0e-2)

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=0.0)

    g0 = ag.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    inversion = plane_to_inversion.inversion_imaging_from(
        dataset=masked_imaging_7x7,
        image=masked_imaging_7x7.image,
        noise_map=masked_imaging_7x7.noise_map,
        w_tilde=masked_imaging_7x7.w_tilde,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert inversion.mapped_reconstructed_image == pytest.approx(
        masked_imaging_7x7.image, 1.0e-2
    )


def test__inversion_interferometer_from(sub_grid_2d_7x7, interferometer_7):

    g_linear = ag.Galaxy(redshift=0.5, light_linear=ag.lp_linear.EllSersic())

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g_linear])

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    inversion = plane_to_inversion.inversion_interferometer_from(
        dataset=interferometer_7,
        visibilities=interferometer_7.visibilities,
        noise_map=interferometer_7.noise_map,
        w_tilde=None,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(
            use_w_tilde=False, use_linear_operators=False
        ),
    )

    assert inversion.reconstruction[0] == pytest.approx(0.0012073, 1.0e-2)

    interferometer_7.data = ag.Visibilities.ones(shape_slim=(7,))

    pix = ag.pix.Rectangular(shape=(7, 7))
    reg = ag.reg.Constant(coefficient=0.0)

    g0 = ag.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    plane_to_inversion = ag.PlaneToInversion(plane=plane)

    inversion = plane_to_inversion.inversion_interferometer_from(
        dataset=interferometer_7,
        visibilities=interferometer_7.visibilities,
        noise_map=interferometer_7.noise_map,
        w_tilde=None,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(
            use_w_tilde=False, use_linear_operators=False
        ),
    )

    assert inversion.mapped_reconstructed_data.real == pytest.approx(
        interferometer_7.visibilities.real, 1.0e-2
    )
