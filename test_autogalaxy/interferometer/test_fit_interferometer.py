import numpy as np
import pytest

import autogalaxy as ag


def test__model_visibilities(interferometer_7):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.m.MockLightProfile(image_2d=np.ones(9)))

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=[g0])

    assert fit.model_data.slim[0].real == pytest.approx(1.48496, abs=1.0e-4)
    assert fit.model_data.slim[0].imag == pytest.approx(0.0, abs=1.0e-4)
    assert fit.log_likelihood == pytest.approx(-34.1685958, abs=1.0e-4)


def test__fit_figure_of_merit(interferometer_7):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)))

    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)))

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=[g0, g1])

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-1994.35383952, 1.0e-4)

    basis = ag.lp_basis.Basis(
        profile_list=[
            ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)),
            ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)),
        ]
    )

    g0 = ag.Galaxy(redshift=0.5, bulge=basis)

    fit = ag.FitInterferometer(dataset=interferometer_7, galaxies=[g0])

    assert fit.perform_inversion is False
    assert fit.figure_of_merit == pytest.approx(-1994.3538395, 1.0e-4)

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=0.01),
    )

    g0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[ag.Galaxy(redshift=0.5), g0],
    )

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-71.770448724198, 1.0e-4)

    galaxy_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05))
    )

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[galaxy_light, galaxy_pix],
    )

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-196.15073725528504, 1.0e-4)

    g0_linear_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear.Sersic(sersic_index=1.0, centre=(0.05, 0.05))
    )

    g1_linear_light = ag.Galaxy(
        redshift=0.5, bulge=ag.lp_linear.Sersic(sersic_index=4.0, centre=(0.05, 0.05))
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=[g0_linear_light, g1_linear_light]
    )

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-23.44419, 1.0e-4)

    basis = ag.lp_basis.Basis(
        profile_list=[
            ag.lp_linear.Sersic(sersic_index=1.0, centre=(0.05, 0.05)),
            ag.lp_linear.Sersic(sersic_index=4.0, centre=(0.05, 0.05)),
        ]
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=[ag.Galaxy(redshift=0.5, bulge=basis)]
    )

    assert fit.perform_inversion is True
    assert fit.figure_of_merit == pytest.approx(-23.44419235, 1.0e-4)

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=[g0_linear_light, galaxy_pix]
    )

    assert fit.log_evidence == pytest.approx(-37.4081355120388, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-37.4081355120388, 1.0e-4)


def test___galaxy_model_image_dict(interferometer_7):
    # Normal Light Profiles Only

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)))
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)))
    g2 = ag.Galaxy(
        redshift=0.5,
        light_profile_0=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)),
        light_profile_1=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)),
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, g1, g2],
    )

    g0_image = g0.image_2d_from(grid=interferometer_7.grids.lp)
    g1_image = g1.image_2d_from(grid=interferometer_7.grids.lp)

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(g0_image.array, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1] == pytest.approx(g1_image.array, 1.0e-4)
    assert fit.galaxy_model_image_dict[g2] == pytest.approx(
        g0_image.array + g1_image.array, 1.0e-4
    )

    # Linear Light Profiles Only

    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05)))

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear],
    )

    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(
        0.9876689631, 1.0e-4
    )

    # Pixelization + Regularizaiton only

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(redshift=0.5)
    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, galaxy_pix_0],
        settings_inversion=ag.SettingsInversion(use_border_relocator=True),
    )

    mapper_grids = pixelization.mesh.mapper_grids_from(
        mask=interferometer_7.real_space_mask,
        source_plane_data_grid=interferometer_7.grids.lp,
        border_relocator=interferometer_7.grids.border_relocator,
        source_plane_mesh_grid=None,
    )

    mapper = ag.Mapper(
        mapper_grids=mapper_grids,
        border_relocator=interferometer_7.grids.border_relocator,
        regularization=pixelization.regularization,
    )

    inversion = ag.Inversion(
        dataset=interferometer_7,
        linear_obj_list=[mapper],
    )

    assert (fit.galaxy_model_image_dict[g0].native == 0.0 + 0.0j * np.zeros((7,))).all()

    assert fit.galaxy_model_image_dict[galaxy_pix_0].array == pytest.approx(
        inversion.mapped_reconstructed_image.slim.array, 1.0e-4
    )

    # Linear Light PRofiles + Pixelization + Regularizaiton

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=2.0),
    )

    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear, g1, galaxy_pix_0, galaxy_pix_1],
    )

    assert fit.galaxy_model_image_dict[g0_linear][4] == pytest.approx(
        -46.8820117, 1.0e-2
    )
    assert fit.galaxy_model_image_dict[g1] == pytest.approx(g1_image.array, 1.0e-4)
    assert fit.galaxy_model_image_dict[galaxy_pix_0][4] == pytest.approx(
        -0.00541699, 1.0e-2
    )
    assert fit.galaxy_model_image_dict[galaxy_pix_1][4] == pytest.approx(
        -0.00563034, 1.0e-2
    )

    mapped_reconstructed_image = (
        fit.galaxy_model_image_dict[g0_linear]
        + fit.galaxy_model_image_dict[galaxy_pix_0]
        + fit.galaxy_model_image_dict[galaxy_pix_1]
    )

    assert mapped_reconstructed_image.array == pytest.approx(
        fit.inversion.mapped_reconstructed_image.array, 1.0e-4
    )


def test___galaxy_model_visibilities_dict(interferometer_7):
    # Normal Light Profiles Only

    g0 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)))
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)))
    g2 = ag.Galaxy(
        redshift=0.5,
        light_profile_0=ag.lp.Sersic(intensity=1.0, centre=(0.05, 0.05)),
        light_profile_1=ag.lp.Sersic(intensity=2.0, centre=(0.05, 0.05)),
    )

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, g1, g2],
    )

    g0_visibilities = g0.visibilities_from(
        grid=interferometer_7.grids.lp, transformer=interferometer_7.transformer
    )
    g1_visibilities = g1.visibilities_from(
        grid=interferometer_7.grids.lp, transformer=interferometer_7.transformer
    )

    assert fit.galaxy_model_visibilities_dict[g0].array == pytest.approx(
        g0_visibilities.array, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g1].array == pytest.approx(
        g1_visibilities.array, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g2].array == pytest.approx(
        g0_visibilities.array + g1_visibilities.array, 1.0e-4
    )

    # Linear Light Profiles Only

    g0_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic(centre=(0.05, 0.05)))

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear],
    )

    assert fit.galaxy_model_visibilities_dict[g0_linear][0] == pytest.approx(
        0.9965209248910107 + 0.00648675263899049j, 1.0e-4
    )

    # Pixelization + Regularizaiton only

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(redshift=0.5)
    galaxy_pix_0 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0, galaxy_pix_0],
        settings_inversion=ag.SettingsInversion(use_border_relocator=True),
    )

    mapper_grids = pixelization.mesh.mapper_grids_from(
        mask=interferometer_7.real_space_mask,
        source_plane_data_grid=interferometer_7.grids.lp,
        border_relocator=interferometer_7.grids.border_relocator,
        source_plane_mesh_grid=None,
    )

    mapper = ag.Mapper(
        mapper_grids=mapper_grids,
        border_relocator=interferometer_7.grids.border_relocator,
        regularization=pixelization.regularization,
    )

    inversion = ag.Inversion(
        dataset=interferometer_7,
        linear_obj_list=[mapper],
    )

    assert (fit.galaxy_model_visibilities_dict[g0] == 0.0 + 0.0j * np.zeros((7,))).all()

    assert fit.galaxy_model_visibilities_dict[galaxy_pix_0].array == pytest.approx(
        inversion.mapped_reconstructed_data.array, 1.0e-4
    )

    # Linear Light PRofiles + Pixelization + Regularizaiton
    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=2.0),
    )

    galaxy_pix_1 = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        galaxies=[g0_linear, g1, galaxy_pix_0, galaxy_pix_1],
    )

    assert fit.galaxy_model_visibilities_dict[g0_linear][0] == pytest.approx(
        -47.30219078770512 - 0.3079088489343429j, 1.0e-4
    )

    assert fit.galaxy_model_visibilities_dict[g1].array == pytest.approx(
        g1_visibilities.array, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[galaxy_pix_0][0] == pytest.approx(
        -0.00889895 + 0.22151583j, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[galaxy_pix_1][0] == pytest.approx(
        -0.00857457 + 0.05537896j, 1.0e-4
    )

    mapped_reconstructed_visibilities = (
        fit.galaxy_model_visibilities_dict[g0_linear]
        + fit.galaxy_model_visibilities_dict[galaxy_pix_0]
        + fit.galaxy_model_visibilities_dict[galaxy_pix_1]
    )

    assert mapped_reconstructed_visibilities.array == pytest.approx(
        fit.inversion.mapped_reconstructed_data.array, 1.0e-4
    )


def test__model_visibilities_of_galaxies_list(interferometer_7):
    galaxy_light = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))
    galaxy_linear = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.Sersic())

    pixelization = ag.Pixelization(
        mesh=ag.mesh.RectangularUniform(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    fit = ag.FitInterferometer(
        dataset=interferometer_7, galaxies=[galaxy_light, galaxy_linear, galaxy_pix]
    )

    assert fit.model_visibilities_of_galaxies_list[0].array == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_light].array, 1.0e-4
    )
    assert fit.model_visibilities_of_galaxies_list[1].array == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_linear].array, 1.0e-4
    )
    assert fit.model_visibilities_of_galaxies_list[2].array == pytest.approx(
        fit.galaxy_model_visibilities_dict[galaxy_pix].array, 1.0e-4
    )
