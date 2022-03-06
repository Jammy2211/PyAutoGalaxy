import numpy as np
import pytest

import autogalaxy as ag


def test__model_image__with_and_without_psf_blurring(
    masked_imaging_7x7_no_blur, masked_imaging_7x7
):

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.m.MockLightProfile(
            image_2d_value=1.0, image_2d_first_value=2.0
        ),
    )
    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.model_image.slim == pytest.approx(
        np.array([2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-4
    )
    assert fit.log_likelihood == pytest.approx(-14.63377, 1.0e-4)

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert fit.model_image.slim == pytest.approx(
        np.array([1.33, 1.16, 1.0, 1.16, 1.0, 1.0, 1.0, 1.0, 1.0]), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-14.52960, 1.0e-4)


def test__noise_map__with_and_without_hyper_galaxy(masked_imaging_7x7_no_blur):

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.m.MockLightProfile(image_2d_value=1.0)
    )

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    hyper_image = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.m.MockLightProfile(image_2d_value=1.0),
        hyper_galaxy=ag.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        ),
        hyper_model_image=hyper_image,
        hyper_galaxy_image=hyper_image,
        hyper_minimum_value=0.0,
    )

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=4.0, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-20.7470, 1.0e-4)


def test__noise_map__with_hyper_galaxy_reaches_upper_limit(masked_imaging_7x7_no_blur):

    hyper_image = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.m.MockLightProfile(image_2d_value=1.0),
        hyper_galaxy=ag.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0e9, noise_power=1.0
        ),
        hyper_model_image=hyper_image,
        hyper_galaxy_image=hyper_image,
        hyper_minimum_value=0.0,
    )

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=1.0e8, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-174.0565, 1.0e-4)


def test__image__with_and_without_hyper_background_sky(masked_imaging_7x7_no_blur):

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.m.MockLightProfile(image_2d_value=1.0)
    )

    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.image.slim == pytest.approx(np.full(fill_value=1.0, shape=(9,)), 1.0e-1)

    hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7_no_blur, plane=plane, hyper_image_sky=hyper_image_sky
    )

    assert fit.image.slim == pytest.approx(np.full(fill_value=2.0, shape=(9,)), 1.0e-1)
    assert fit.log_likelihood == pytest.approx(-15.6337, 1.0e-4)


def test__noise_map__with_and_without_hyper_background(masked_imaging_7x7_no_blur):

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.m.MockLightProfile(image_2d_value=1.0)
    )
    plane = ag.Plane(galaxies=[g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7_no_blur,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
    )

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=3.0, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-18.1579, 1.0e-4)


def test__fit_figure_of_merit(masked_imaging_7x7):

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.lp.EllSersic(intensity=1.0),
        mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
    )

    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert (fit.image == np.full(fill_value=1.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=2.0, shape=(9,))).all()
    assert fit.log_likelihood == pytest.approx(-75938.05, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-75938.05, 1.0e-4)

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)

    g0 = ag.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert (fit.image == np.full(fill_value=1.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=2.0, shape=(9,))).all()
    assert fit.log_evidence == pytest.approx(-22.9005, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-22.9005, 1.0e-4)

    galaxy_light = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)
    galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[galaxy_light, galaxy_pix])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    assert (fit.image == np.full(fill_value=1.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=2.0, shape=(9,))).all()
    assert fit.log_evidence == pytest.approx(-6840.5851, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-6840.5851, 1.0e-4)


def test__fit_figure_of_merit__include_hyper_methods(masked_imaging_7x7):

    hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)
    hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    hyper_galaxy = ag.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
    )

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.lp.EllSersic(intensity=1.0),
        mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=np.ones(9),
        hyper_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )
    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.log_likelihood == pytest.approx(-12104.68, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-12104.68, 1.0e-4)

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)

    g0 = ag.Galaxy(
        redshift=0.5,
        pixelization=pix,
        regularization=reg,
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=np.ones(9),
        hyper_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.log_evidence == pytest.approx(-30.1448, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-30.1448, 1.0e-4)

    galaxy_light = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.lp.EllSersic(intensity=1.0),
        hyper_galaxy=hyper_galaxy,
        hyper_model_image=np.ones(9),
        hyper_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )

    galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[galaxy_light, galaxy_pix])

    fit = ag.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.log_evidence == pytest.approx(-1132.9297, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-1132.9297, 1.0e-4)


def test__galaxy_model_image_dict(masked_imaging_7x7):

    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.lp.EllSersic(intensity=1.0),
        mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
    )
    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))
    g2 = ag.Galaxy(redshift=1.0)

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1, g2])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    g0_blurred_image_2d = g0.blurred_image_2d_via_convolver_from(
        grid=masked_imaging_7x7.grid,
        blurring_grid=masked_imaging_7x7.blurring_grid,
        convolver=masked_imaging_7x7.convolver,
    )

    g1_blurred_image_2d = g1.blurred_image_2d_via_convolver_from(
        grid=masked_imaging_7x7.grid,
        blurring_grid=masked_imaging_7x7.blurring_grid,
        convolver=masked_imaging_7x7.convolver,
    )

    assert fit.galaxy_model_image_dict[g0] == pytest.approx(g0_blurred_image_2d, 1.0e-4)
    assert fit.galaxy_model_image_dict[g1] == pytest.approx(g1_blurred_image_2d, 1.0e-4)
    assert (fit.galaxy_model_image_dict[g2].slim == np.zeros(9)).all()

    assert fit.model_image.native == pytest.approx(
        fit.galaxy_model_image_dict[g0].native + fit.galaxy_model_image_dict[g1].native,
        1.0e-4,
    )

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)

    g0 = ag.Galaxy(redshift=0.5)
    g1 = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    mapper = pix.mapper_from(
        source_grid_slim=masked_imaging_7x7.grid, source_pixelization_grid=None
    )

    inversion = ag.Inversion(
        dataset=masked_imaging_7x7, linear_obj_list=[mapper], regularization_list=[reg]
    )

    assert (fit.galaxy_model_image_dict[g0] == np.zeros(9)).all()

    assert fit.galaxy_model_image_dict[g1].native == pytest.approx(
        inversion.mapped_reconstructed_image.native, 1.0e-4
    )

    assert fit.model_image.native == pytest.approx(
        fit.galaxy_model_image_dict[g1].native, 1.0e-4
    )

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=2.0))
    g2 = ag.Galaxy(redshift=0.5)

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)
    galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1, g2, galaxy_pix])

    masked_imaging_7x7.image[0] = 3.0

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    g0_blurred_image = g0.blurred_image_2d_via_convolver_from(
        grid=masked_imaging_7x7.grid,
        convolver=masked_imaging_7x7.convolver,
        blurring_grid=masked_imaging_7x7.blurring_grid,
    )
    g1_blurred_image = g1.blurred_image_2d_via_convolver_from(
        grid=masked_imaging_7x7.grid,
        convolver=masked_imaging_7x7.convolver,
        blurring_grid=masked_imaging_7x7.blurring_grid,
    )

    blurred_image = g0_blurred_image + g1_blurred_image

    profile_subtracted_image = masked_imaging_7x7.image - blurred_image
    mapper = pix.mapper_from(
        source_grid_slim=masked_imaging_7x7.grid,
        settings=ag.SettingsPixelization(use_border=False),
    )

    inversion = ag.InversionImaging(
        image=profile_subtracted_image,
        noise_map=masked_imaging_7x7.noise_map,
        convolver=masked_imaging_7x7.convolver,
        w_tilde=masked_imaging_7x7.w_tilde,
        linear_obj_list=[mapper],
        regularization_list=[reg],
    )

    assert (fit.galaxy_model_image_dict[g2] == np.zeros(9)).all()

    assert fit.galaxy_model_image_dict[g0].native == pytest.approx(
        g0_blurred_image.native, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[g1].native == pytest.approx(
        g1_blurred_image.native, 1.0e-4
    )
    assert fit.galaxy_model_image_dict[galaxy_pix].native == pytest.approx(
        inversion.mapped_reconstructed_image.native, 1.0e-4
    )

    assert fit.model_image.native == pytest.approx(
        fit.galaxy_model_image_dict[g0].native
        + fit.galaxy_model_image_dict[g1].native
        + inversion.mapped_reconstructed_image.native,
        1.0e-4,
    )


def test___blurred_and_model_image_properties(masked_imaging_7x7):
    g0 = ag.Galaxy(
        redshift=0.5,
        light_profile=ag.lp.EllSersic(intensity=1.0),
        mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
    )

    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=1.0))

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    g0_blurred_image = g0.blurred_image_2d_via_convolver_from(
        grid=masked_imaging_7x7.grid,
        convolver=masked_imaging_7x7.convolver,
        blurring_grid=masked_imaging_7x7.blurring_grid,
    )
    g1_blurred_image = g1.blurred_image_2d_via_convolver_from(
        grid=masked_imaging_7x7.grid,
        convolver=masked_imaging_7x7.convolver,
        blurring_grid=masked_imaging_7x7.blurring_grid,
    )

    assert g0_blurred_image.native == pytest.approx(
        fit.model_images_of_galaxies[0].native, 1.0e-4
    )

    assert g1_blurred_image.native == pytest.approx(
        fit.model_images_of_galaxies[1].native, 1.0e-4
    )

    unmasked_blurred_image = plane.unmasked_blurred_image_2d_via_psf_from(
        grid=masked_imaging_7x7.grid, psf=masked_imaging_7x7.psf
    )

    assert (unmasked_blurred_image == fit.unmasked_blurred_image).all()

    unmasked_blurred_image_of_galaxies = plane.unmasked_blurred_image_2d_list_via_psf_from(
        grid=masked_imaging_7x7.grid, psf=masked_imaging_7x7.psf
    )

    assert (
        unmasked_blurred_image_of_galaxies[0]
        == fit.unmasked_blurred_image_of_galaxies[0]
    ).all()
    assert (
        unmasked_blurred_image_of_galaxies[1]
        == fit.unmasked_blurred_image_of_galaxies[1]
    ).all()

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)

    g0 = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[ag.Galaxy(redshift=0.5), g0])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    mapper = pix.mapper_from(
        source_grid_slim=masked_imaging_7x7.grid,
        settings=ag.SettingsPixelization(use_border=False),
    )

    inversion = ag.Inversion(
        dataset=masked_imaging_7x7, linear_obj_list=[mapper], regularization_list=[reg]
    )

    assert (fit.model_images_of_galaxies[0].native == np.zeros((7, 7))).all()
    assert inversion.mapped_reconstructed_image.native == pytest.approx(
        fit.model_images_of_galaxies[1].native, 1.0e-4
    )

    galaxy_light = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))

    pix = ag.pix.Rectangular(shape=(3, 3))
    reg = ag.reg.Constant(coefficient=1.0)
    galaxy_pix = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(redshift=0.75, galaxies=[galaxy_light, galaxy_pix])

    fit = ag.FitImaging(dataset=masked_imaging_7x7, plane=plane)

    blurred_image = plane.blurred_image_2d_via_convolver_from(
        grid=masked_imaging_7x7.grid,
        convolver=masked_imaging_7x7.convolver,
        blurring_grid=masked_imaging_7x7.blurring_grid,
    )

    profile_subtracted_image = masked_imaging_7x7.image - blurred_image

    mapper = pix.mapper_from(
        source_grid_slim=masked_imaging_7x7.grid,
        settings=ag.SettingsPixelization(use_border=False),
    )

    inversion = ag.InversionImaging(
        image=profile_subtracted_image,
        noise_map=masked_imaging_7x7.noise_map,
        convolver=masked_imaging_7x7.convolver,
        w_tilde=masked_imaging_7x7.w_tilde,
        linear_obj_list=[mapper],
        regularization_list=[reg],
    )

    assert blurred_image.native == pytest.approx(
        fit.model_images_of_galaxies[0].native, 1.0e-4
    )
    assert inversion.mapped_reconstructed_image.native == pytest.approx(
        fit.model_images_of_galaxies[1].native, 1.0e-4
    )


def test__subtracted_images_of_galaxies(masked_imaging_7x7_no_blur):

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.m.MockLightProfile(image_2d=np.ones(1))
    )

    g1 = ag.Galaxy(
        redshift=1.0, light_profile=ag.m.MockLightProfile(image_2d=2.0 * np.ones(1))
    )

    g2 = ag.Galaxy(
        redshift=1.0, light_profile=ag.m.MockLightProfile(image_2d=3.0 * np.ones(1))
    )

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1, g2])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.subtracted_images_of_galaxies[0].slim[0] == -4.0
    assert fit.subtracted_images_of_galaxies[1].slim[0] == -3.0
    assert fit.subtracted_images_of_galaxies[2].slim[0] == -2.0

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.m.MockLightProfile(image_2d=np.ones(1))
    )

    g1 = ag.Galaxy(redshift=0.5)

    g2 = ag.Galaxy(
        redshift=1.0, light_profile=ag.m.MockLightProfile(image_2d=3.0 * np.ones(1))
    )

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1, g2])

    fit = ag.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.subtracted_images_of_galaxies[0].slim[0] == -2.0
    assert fit.subtracted_images_of_galaxies[1].slim[0] == -3.0
    assert fit.subtracted_images_of_galaxies[2].slim[0] == 0.0
