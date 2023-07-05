import numpy as np
import pytest

import autogalaxy as ag


def test__lp_linear_func_list_galaxy_dict(lp_0, masked_imaging_7x7):
    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    lp_linear_func_galaxy_dict = plane_to_inversion.lp_linear_func_list_galaxy_dict

    assert lp_linear_func_galaxy_dict == {}

    lp_linear_0 = ag.lp_linear.LightProfileLinear()
    lp_linear_1 = ag.lp_linear.LightProfileLinear()
    lp_linear_2 = ag.lp_linear.LightProfileLinear()

    galaxy_0 = ag.Galaxy(
        redshift=0.5, lp_0=lp_0, light_linear_0=lp_linear_0, light_linear_1=lp_linear_1
    )

    galaxy_1 = ag.Galaxy(redshift=0.5, light_linear=lp_linear_2)

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1], redshift=None)

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    lp_linear_func_galaxy_dict = plane_to_inversion.lp_linear_func_list_galaxy_dict

    lp_linear_func_list = list(lp_linear_func_galaxy_dict.keys())

    assert lp_linear_func_galaxy_dict[lp_linear_func_list[0]] == galaxy_0
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[1]] == galaxy_0
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[2]] == galaxy_1

    assert lp_linear_func_list[0].light_profile_list[0] == lp_linear_0
    assert lp_linear_func_list[1].light_profile_list[0] == lp_linear_1
    assert lp_linear_func_list[2].light_profile_list[0] == lp_linear_2

    basis = ag.lp_basis.Basis(light_profile_list=[lp_linear_0, lp_linear_1])

    galaxy_0 = ag.Galaxy(redshift=0.5, bulge=basis)

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1], redshift=None)

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    lp_linear_func_galaxy_dict = plane_to_inversion.lp_linear_func_list_galaxy_dict

    lp_linear_func_list = list(lp_linear_func_galaxy_dict.keys())

    assert lp_linear_func_galaxy_dict[lp_linear_func_list[0]] == galaxy_1
    assert lp_linear_func_galaxy_dict[lp_linear_func_list[1]] == galaxy_0

    assert lp_linear_func_list[0].light_profile_list[0] == lp_linear_2
    assert lp_linear_func_list[1].light_profile_list[0] == lp_linear_0
    assert lp_linear_func_list[1].light_profile_list[1] == lp_linear_1


def test__sparse_image_plane_grid_list(masked_imaging_7x7):
    pixelization = ag.m.MockPixelization(
        mesh=ag.m.MockMesh(image_plane_mesh_grid=np.array([1.0, 1.0]))
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.Plane(galaxies=[galaxy_pix], redshift=0.5)

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    sparse_image_plane_grid_list = plane_to_inversion.sparse_image_plane_grid_list

    assert (sparse_image_plane_grid_list == np.array([[1.0, 1.0]])).all()

    # In the ag.m.MockPixelization class the grid is returned if hyper image=None, and grid*hyper image is
    # returned otherwise.

    galaxy_pix = ag.Galaxy(
        redshift=0.5, pixelization=pixelization, adapt_galaxy_image=2
    )

    plane = ag.Plane(galaxies=[galaxy_pix], redshift=0.5)

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    sparse_image_plane_grid_list = plane_to_inversion.sparse_image_plane_grid_list

    assert (sparse_image_plane_grid_list == np.array([[2.0, 2.0]])).all()

    # No Galalxies

    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(galaxies=[galaxy_no_pix], redshift=0.5)

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    sparse_image_plane_grid_list = plane_to_inversion.sparse_image_plane_grid_list

    assert sparse_image_plane_grid_list is None


def test__mapper_galaxy_dict(masked_imaging_7x7):
    mesh = ag.mesh.Rectangular(shape=(3, 3))

    pixelization = ag.m.MockPixelization(mesh=mesh)

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)
    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(galaxies=[galaxy_no_pix, galaxy_pix], redshift=0.5)

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    mapper_galaxy_dict = plane_to_inversion.mapper_galaxy_dict

    mapper_list = list(mapper_galaxy_dict.keys())

    assert mapper_list[0].pixels == 9
    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix

    mesh = ag.mesh.Rectangular(shape=(4, 3))
    pixelization = ag.m.MockPixelization(mesh=mesh)

    galaxy_pix_2 = ag.Galaxy(redshift=0.5, pixelization=pixelization)
    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(
        galaxies=[galaxy_no_pix, galaxy_pix, galaxy_no_pix, galaxy_pix_2], redshift=0.5
    )

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    mapper_galaxy_dict = plane_to_inversion.mapper_galaxy_dict

    mapper_list = list(mapper_galaxy_dict.keys())

    assert mapper_list[0].pixels == 9
    assert mapper_list[1].pixels == 12

    assert mapper_galaxy_dict[mapper_list[0]] == galaxy_pix
    assert mapper_galaxy_dict[mapper_list[1]] == galaxy_pix_2

    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(galaxies=[galaxy_no_pix], redshift=0.5)

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    mapper_galaxy_dict = plane_to_inversion.mapper_galaxy_dict

    assert mapper_galaxy_dict == {}


def test__regularization_list(masked_imaging_7x7):
    regularization_0 = ag.reg.Constant(coefficient=1.0)
    regularization_1 = ag.reg.ConstantSplit(coefficient=2.0)

    pixelization_0 = ag.m.MockPixelization(
        mesh=ag.mesh.Rectangular(shape=(10, 10)), regularization=regularization_0
    )
    pixelization_1 = ag.m.MockPixelization(
        mesh=ag.mesh.Rectangular(shape=(8, 8)), regularization=regularization_1
    )

    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp_linear.Gaussian())
    galaxy_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization_0)
    galaxy_2 = ag.Galaxy(
        redshift=0.5, light=ag.lp_linear.Gaussian(), pixelization=pixelization_1
    )

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    regularization_list = plane_to_inversion.regularization_list

    assert regularization_list[0] == None
    assert regularization_list[1] == None
    assert regularization_list[2] == regularization_0
    assert regularization_list[3] == regularization_1

    regularization_2 = ag.reg.Constant(coefficient=3.0)

    basis = ag.lp_basis.Basis(
        light_profile_list=[ag.lp_linear.Gaussian()], regularization=regularization_2
    )

    galaxy_3 = ag.Galaxy(redshift=0.5, bulge=basis)

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1, galaxy_3])

    plane_to_inversion = ag.PlaneToInversion(plane=plane, dataset=masked_imaging_7x7)

    regularization_list = plane_to_inversion.regularization_list

    assert regularization_list[0] == None
    assert regularization_list[1] == regularization_2
    assert regularization_list[2] == regularization_0


def test__inversion_imaging_from(sub_grid_2d_7x7, masked_imaging_7x7):
    g_linear = ag.Galaxy(redshift=0.5, light_linear=ag.lp_linear.Sersic())

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g_linear])

    plane_to_inversion = ag.PlaneToInversion(
        plane=plane,
        dataset=masked_imaging_7x7,
        data=masked_imaging_7x7.data,
        noise_map=masked_imaging_7x7.noise_map,
        w_tilde=masked_imaging_7x7.w_tilde,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    inversion = plane_to_inversion.inversion

    assert inversion.reconstruction[0] == pytest.approx(0.00543437, 1.0e-2)

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=0.0),
    )

    g0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    plane_to_inversion = ag.PlaneToInversion(
        plane=plane,
        dataset=masked_imaging_7x7,
        data=masked_imaging_7x7.data,
        noise_map=masked_imaging_7x7.noise_map,
        w_tilde=masked_imaging_7x7.w_tilde,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    inversion = plane_to_inversion.inversion

    assert inversion.mapped_reconstructed_image == pytest.approx(
        masked_imaging_7x7.data, 1.0e-2
    )


def test__inversion_interferometer_from(sub_grid_2d_7x7, interferometer_7):
    g_linear = ag.Galaxy(redshift=0.5, light_linear=ag.lp_linear.Sersic())

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g_linear])

    plane_to_inversion = ag.PlaneToInversion(
        plane=plane,
        dataset=interferometer_7,
        data=interferometer_7.visibilities,
        noise_map=interferometer_7.noise_map,
        w_tilde=None,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(
            use_w_tilde=False, use_linear_operators=False
        ),
    )

    inversion = plane_to_inversion.inversion

    assert inversion.reconstruction[0] == pytest.approx(0.0012073, 1.0e-2)

    interferometer_7.data = ag.Visibilities.ones(shape_slim=(7,))

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(7, 7)),
        regularization=ag.reg.Constant(coefficient=0.0),
    )

    g0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    plane_to_inversion = ag.PlaneToInversion(
        plane=plane,
        dataset=interferometer_7,
        data=interferometer_7.visibilities,
        noise_map=interferometer_7.noise_map,
        w_tilde=None,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(
            use_w_tilde=False, use_linear_operators=False
        ),
    )

    inversion = plane_to_inversion.inversion

    assert inversion.mapped_reconstructed_data.real == pytest.approx(
        interferometer_7.visibilities.real, 1.0e-2
    )


def test__raises_exception_if_noise_covariance_input(masked_imaging_covariance_7x7):
    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

    with pytest.raises(ag.exc.InversionException):
        ag.PlaneToInversion(plane=plane, dataset=masked_imaging_covariance_7x7)
