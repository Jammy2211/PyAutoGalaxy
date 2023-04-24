import numpy as np
import pytest

import autogalaxy as ag


def test__noise_map__with_hyper_galaxy(masked_imaging_7x7_no_blur):
    hyper_image = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.m.MockLightProfile(image_2d_value=1.0),
        hyper_galaxy=ag.legacy.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        ),
        adapt_model_image=hyper_image,
        adapt_galaxy_image=hyper_image,
        hyper_minimum_value=0.0,
    )

    plane = ag.legacy.Plane(galaxies=[g0])

    fit = ag.legacy.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=4.0, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-20.7470, 1.0e-4)


def test__noise_map__with_hyper_galaxy_reaches_upper_limit(masked_imaging_7x7_no_blur):
    hyper_image = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0)

    g0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.m.MockLightProfile(image_2d_value=1.0),
        hyper_galaxy=ag.legacy.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0e9, noise_power=1.0
        ),
        adapt_model_image=hyper_image,
        adapt_galaxy_image=hyper_image,
        hyper_minimum_value=0.0,
    )

    plane = ag.legacy.Plane(galaxies=[g0])

    fit = ag.legacy.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=1.0e8, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-174.0565, 1.0e-4)


def test__image__with_and_without_hyper_background_sky(masked_imaging_7x7_no_blur):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.m.MockLightProfile(image_2d_value=1.0))

    plane = ag.legacy.Plane(galaxies=[g0])

    fit = ag.legacy.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.image.slim == pytest.approx(np.full(fill_value=1.0, shape=(9,)), 1.0e-1)

    hyper_image_sky = ag.legacy.hyper_data.HyperImageSky(sky_scale=1.0)

    fit = ag.legacy.FitImaging(
        dataset=masked_imaging_7x7_no_blur, plane=plane, hyper_image_sky=hyper_image_sky
    )

    assert fit.image.slim == pytest.approx(np.full(fill_value=2.0, shape=(9,)), 1.0e-1)
    assert fit.log_likelihood == pytest.approx(-15.6337, 1.0e-4)


def test__noise_map__with_and_without_hyper_background(masked_imaging_7x7_no_blur):
    g0 = ag.Galaxy(redshift=0.5, bulge=ag.m.MockLightProfile(image_2d_value=1.0))
    plane = ag.legacy.Plane(galaxies=[g0])

    fit = ag.legacy.FitImaging(dataset=masked_imaging_7x7_no_blur, plane=plane)

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=2.0, shape=(9,)), 1.0e-1
    )

    hyper_background_noise = ag.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    fit = ag.legacy.FitImaging(
        dataset=masked_imaging_7x7_no_blur,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
    )

    assert fit.noise_map.slim == pytest.approx(
        np.full(fill_value=3.0, shape=(9,)), 1.0e-1
    )
    assert fit.log_likelihood == pytest.approx(-18.1579, 1.0e-4)


def test__fit_figure_of_merit__include_hyper_methods(masked_imaging_7x7):
    hyper_image_sky = ag.legacy.hyper_data.HyperImageSky(sky_scale=1.0)
    hyper_background_noise = ag.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    hyper_galaxy = ag.legacy.HyperGalaxy(
        contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
    )

    g0 = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(intensity=1.0),
        hyper_galaxy=hyper_galaxy,
        adapt_model_image=np.ones(9),
        adapt_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )
    g1 = ag.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[g0, g1])

    fit = ag.legacy.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.figure_of_merit == pytest.approx(-12104.68, 1.0e-4)

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    g0 = ag.Galaxy(
        redshift=0.5,
        pixelization=pixelization,
        hyper_galaxy=hyper_galaxy,
        adapt_model_image=np.ones(9),
        adapt_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )

    plane = ag.legacy.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

    fit = ag.legacy.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.figure_of_merit == pytest.approx(-30.1448, 1.0e-4)

    galaxy_light = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.Sersic(intensity=1.0),
        hyper_galaxy=hyper_galaxy,
        adapt_model_image=np.ones(9),
        adapt_galaxy_image=np.ones(9),
        hyper_minimum_value=0.0,
    )

    galaxy_pix = ag.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[galaxy_light, galaxy_pix])

    fit = ag.legacy.FitImaging(
        dataset=masked_imaging_7x7,
        plane=plane,
        hyper_image_sky=hyper_image_sky,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.image == np.full(fill_value=2.0, shape=(9,))).all()
    assert (fit.noise_map == np.full(fill_value=5.0, shape=(9,))).all()
    assert fit.figure_of_merit == pytest.approx(-1132.9297, 1.0e-4)
