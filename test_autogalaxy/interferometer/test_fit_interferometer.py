import numpy as np
import pytest

import autogalaxy as ag


def test__model_visibilities(interferometer_7):

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.m.MockLightProfile(image_2d=np.ones(2))
    )
    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitInterferometer(dataset=interferometer_7, plane=plane)

    assert fit.model_visibilities.slim[0] == pytest.approx(
        np.array([1.2933 + 0.2829j]), 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(-27.06284, 1.0e-4)


def test__noise_map__with_and_without_hyper_background(interferometer_7):

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.m.MockLightProfile(image_2d=np.ones(2))
    )
    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitInterferometer(dataset=interferometer_7, plane=plane)

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()

    hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_likelihood == pytest.approx(-30.24288, 1.0e-4)


def test__fit_figure_of_merit(interferometer_7):

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.lp.EllSersic(intensity=1.0),
        mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
    )

    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    fit = ag.FitInterferometer(dataset=interferometer_7, plane=plane)

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_likelihood == pytest.approx(-2398107.3849, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-2398107.3849, 1.0e-4)

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=0.01)

    g0 = ag.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-66.90612, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-66.90612, 1.0e-4)

    galaxy_light = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)
    galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[galaxy_light, galaxy_pix])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-283424.48941, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-283424.48941, 1.0e-4)


def test__fit_figure_of_merit__include_hyper_methods(interferometer_7):

    hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.lp.EllSersic(intensity=1.0),
        mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
    )

    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_likelihood == pytest.approx(-1065843.7193, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-1065843.7193, 1.0e-4)

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
        use_hyper_scaling=False,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit.noise_map == pytest.approx(interferometer_7.noise_map, 1.0e-4)

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=0.01)

    g0 = ag.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-68.63380, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-68.63380, 1.0e-4)

    galaxy_light = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)
    galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[galaxy_light, galaxy_pix])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-161108.8377, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-161108.8377, 1.0e-4)


def test___fit_figure_of_merit__different_settings(
    interferometer_7, interferometer_7_lop
):

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=0.01)

    g0 = ag.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        settings_inversion=ag.SettingsInversion(
            use_w_tilde=True, use_linear_operators=False
        ),
    )

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-66.90612, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-66.90612, 1.0e-4)

    fit = ag.FitInterferometer(
        dataset=interferometer_7_lop,
        plane=plane,
        settings_inversion=ag.SettingsInversion(
            use_w_tilde=False, use_linear_operators=True
        ),
    )

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-71.5177, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-71.5177, 1.0e-4)


def test___galaxy_model_image_dict(interferometer_7):

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.lp.EllSersic(intensity=1.0),
        mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
    )
    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    g0_image = g0.image_2d_from(grid=interferometer_7.grid)
    g1_image = g1.image_2d_from(grid=interferometer_7.grid)

    assert fit.galaxy_model_image_dict[g0].slim == pytest.approx(g0_image, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1].slim == pytest.approx(g1_image, 1.0e-4)

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)

    g0 = ag.Galaxy(redshift=0.5)
    g1 = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    mapper = pix.mapper_from(
        source_grid_slim=interferometer_7.grid, source_pixelization_grid=None
    )

    inversion = ag.Inversion(
        dataset=interferometer_7,
        linear_obj_list=[mapper],
        regularization_list=[reg],
        settings=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.galaxy_model_image_dict[g0].native == np.zeros((7, 7))).all()

    assert fit.galaxy_model_image_dict[g1].slim == pytest.approx(
        inversion.mapped_reconstructed_image.slim, 1.0e-4
    )

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=2.0))

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)
    galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1, galaxy_pix])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    g0_visibilities = g0.visibilities_via_transformer_from(
        grid=interferometer_7.grid, transformer=interferometer_7.transformer
    )
    g1_visibilities = g1.visibilities_via_transformer_from(
        grid=interferometer_7.grid, transformer=interferometer_7.transformer
    )

    profile_visibilities = g0_visibilities + g1_visibilities

    profile_subtracted_visibilities = (
        interferometer_7.visibilities - profile_visibilities
    )
    mapper = pix.mapper_from(
        source_grid_slim=interferometer_7.grid,
        settings=ag.SettingsPixelization(use_border=False),
    )

    inversion = ag.InversionInterferometer(
        visibilities=profile_subtracted_visibilities,
        noise_map=interferometer_7.noise_map,
        transformer=interferometer_7.transformer,
        w_tilde=interferometer_7.w_tilde,
        linear_obj_list=[mapper],
        regularization_list=[reg],
        settings=ag.SettingsInversion(use_w_tilde=False),
    )

    g0_image = g0.image_2d_from(grid=interferometer_7.grid)
    g1_image = g1.image_2d_from(grid=interferometer_7.grid)

    assert fit.galaxy_model_image_dict[g0].slim == pytest.approx(g0_image.slim, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1].slim == pytest.approx(g1_image.slim, 1.0e-4)
    assert fit.galaxy_model_image_dict[galaxy_pix].slim == pytest.approx(
        inversion.mapped_reconstructed_image.slim, 1.0e-4
    )


def test___galaxy_model_visibilities_dict(interferometer_7):

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.lp.EllSersic(intensity=1.0),
        mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
    )
    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    g0_visibilities = g0.visibilities_via_transformer_from(
        grid=interferometer_7.grid, transformer=interferometer_7.transformer
    )
    g1_visibilities = g1.visibilities_via_transformer_from(
        grid=interferometer_7.grid, transformer=interferometer_7.transformer
    )

    assert fit.galaxy_model_visibilities_dict[g0].slim == pytest.approx(
        g0_visibilities, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g1].slim == pytest.approx(
        g1_visibilities, 1.0e-4
    )

    assert fit.model_visibilities.slim == pytest.approx(
        fit.galaxy_model_visibilities_dict[g0].slim
        + fit.galaxy_model_visibilities_dict[g1].slim,
        1.0e-4,
    )

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)

    g0 = ag.Galaxy(redshift=0.5)
    g1 = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    mapper = pix.mapper_from(
        source_grid_slim=interferometer_7.grid, source_pixelization_grid=None
    )

    inversion = ag.Inversion(
        dataset=interferometer_7,
        linear_obj_list=[mapper],
        regularization_list=[reg],
        settings=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.galaxy_model_visibilities_dict[g0] == 0.0 + 0.0j * np.zeros((7,))).all()

    assert fit.galaxy_model_visibilities_dict[g1].slim == pytest.approx(
        inversion.mapped_reconstructed_data.slim, 1.0e-4
    )

    assert fit.model_visibilities.slim == pytest.approx(
        fit.galaxy_model_visibilities_dict[g1].slim, 1.0e-4
    )

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=2.0))
    g2 = ag.Galaxy(redshift=0.5)

    galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1, g2, galaxy_pix])

    fit = ag.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    g0_visibilities = g0.visibilities_via_transformer_from(
        grid=interferometer_7.grid, transformer=interferometer_7.transformer
    )
    g1_visibilities = g1.visibilities_via_transformer_from(
        grid=interferometer_7.grid, transformer=interferometer_7.transformer
    )

    profile_visibilities = g0_visibilities + g1_visibilities

    profile_subtracted_visibilities = (
        interferometer_7.visibilities - profile_visibilities
    )
    mapper = pix.mapper_from(
        source_grid_slim=interferometer_7.grid,
        settings=ag.SettingsPixelization(use_border=False),
    )

    inversion = ag.InversionInterferometer(
        visibilities=profile_subtracted_visibilities,
        noise_map=interferometer_7.noise_map,
        transformer=interferometer_7.transformer,
        w_tilde=interferometer_7.w_tilde,
        linear_obj_list=[mapper],
        regularization_list=[reg],
        settings=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.galaxy_model_visibilities_dict[g2] == 0.0 + 0.0j * np.zeros((7,))).all()

    assert fit.galaxy_model_visibilities_dict[g0].slim == pytest.approx(
        g0_visibilities.slim, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[g1].slim == pytest.approx(
        g1_visibilities.slim, 1.0e-4
    )
    assert fit.galaxy_model_visibilities_dict[galaxy_pix].slim == pytest.approx(
        inversion.mapped_reconstructed_data.slim, 1.0e-4
    )

    assert fit.model_visibilities.slim == pytest.approx(
        fit.galaxy_model_visibilities_dict[g0].slim
        + fit.galaxy_model_visibilities_dict[g1].slim
        + inversion.mapped_reconstructed_data.slim,
        1.0e-4,
    )
