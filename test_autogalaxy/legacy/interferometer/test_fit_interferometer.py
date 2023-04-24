import numpy as np
import pytest

import autogalaxy as ag


def test__noise_map__with_and_without_hyper_background(interferometer_7):
    g0 = ag.legacy.Galaxy(
        redshift=0.5, bulge=ag.m.MockLightProfile(image_2d=np.ones(9))
    )
    plane = ag.legacy.Plane(galaxies=[g0])

    fit = ag.legacy.FitInterferometer(dataset=interferometer_7, plane=plane)

    assert (fit.noise_map.slim == np.full(fill_value=2.0 + 2.0j, shape=(7,))).all()

    hyper_background_noise = ag.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    fit = ag.legacy.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_likelihood == pytest.approx(-33.40099, 1.0e-4)


def test__fit_figure_of_merit__include_hyper_methods(interferometer_7):
    hyper_background_noise = ag.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    g0 = ag.legacy.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    g1 = ag.legacy.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[g0, g1])

    fit = ag.legacy.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_likelihood == pytest.approx(-1065843.7193, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-1065843.7193, 1.0e-4)

    fit = ag.legacy.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
        use_hyper_scaling=False,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit.noise_map == pytest.approx(interferometer_7.noise_map, 1.0e-4)

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=0.01),
    )

    g0 = ag.legacy.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.legacy.Plane(galaxies=[ag.legacy.Galaxy(redshift=0.5), g0])

    fit = ag.legacy.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-68.63380, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-68.63380, 1.0e-4)

    galaxy_light = ag.legacy.Galaxy(redshift=0.5, bulge=ag.lp.Sersic(intensity=1.0))

    pixelization = ag.Pixelization(
        mesh=ag.mesh.Rectangular(shape=(3, 3)),
        regularization=ag.reg.Constant(coefficient=1.0),
    )

    galaxy_pix = ag.legacy.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[galaxy_light, galaxy_pix])

    fit = ag.legacy.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert (fit.noise_map.slim == np.full(fill_value=3.0 + 3.0j, shape=(7,))).all()
    assert fit.log_evidence == pytest.approx(-161108.8377, 1e-4)
    assert fit.figure_of_merit == pytest.approx(-161108.8377, 1.0e-4)
