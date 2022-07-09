import os
from os import path
import shutil

import numpy as np
import pytest

import autogalaxy as ag


def test__perfect_fit__chi_squared_0():
    grid = ag.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, sub_size=2)

    galaxy_0 = ag.Galaxy(
        redshift=0.5, light=ag.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1)
    )

    galaxy_1 = ag.Galaxy(
        redshift=0.5, light=ag.lp.EllExponential(centre=(0.1, 0.1), intensity=0.5)
    )

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    simulator = ag.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=ag.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    interferometer = simulator.via_plane_from(plane=plane, grid=grid)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "data_temp",
        "simulate_and_fit",
    )

    try:
        shutil.rmtree(file_path)
    except FileNotFoundError:
        pass

    if path.exists(file_path) is False:
        os.makedirs(file_path)

    interferometer.output_to_fits(
        visibilities_path=path.join(file_path, "visibilities.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(file_path, "uv_wavelengths.fits"),
    )

    real_space_mask = ag.Mask2D.unmasked(
        shape_native=(51, 51), pixel_scales=0.1, sub_size=2
    )

    interferometer = ag.Interferometer.from_fits(
        visibilities_path=path.join(file_path, "visibilities.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(file_path, "uv_wavelengths.fits"),
        real_space_mask=real_space_mask,
        settings=ag.SettingsInterferometer(
            grid_class=ag.Grid2D, transformer_class=ag.TransformerDFT, sub_size=2
        ),
    )

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    fit = ag.FitInterferometer(
        dataset=interferometer,
        plane=plane,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit.chi_squared == pytest.approx(0.0)

    pix = ag.pix.Rectangular(shape=(7, 7))

    reg = ag.reg.Constant(coefficient=0.0001)

    galaxy_0 = ag.Galaxy(
        redshift=0.5, light=ag.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1)
    )

    galaxy_1 = ag.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    fit = ag.FitInterferometer(
        dataset=interferometer,
        plane=plane,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )
    assert abs(fit.chi_squared) < 1.0e-4

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    shutil.rmtree(file_path, ignore_errors=True)


def test__simulate_interferometer_data_and_fit__known_likelihood():

    mask = ag.Mask2D.circular(
        radius=3.0, shape_native=(31, 31), pixel_scales=0.2, sub_size=1
    )

    grid = ag.Grid2D.from_mask(mask=mask)

    galaxy_0 = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=ag.mp.EllIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    galaxy_1 = ag.Galaxy(
        redshift=1.0,
        pixelization=ag.pix.Rectangular(shape=(16, 16)),
        regularization=ag.reg.Constant(coefficient=(1.0)),
    )

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    simulator = ag.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=ag.TransformerDFT,
        exposure_time=300.0,
        noise_seed=1,
    )

    interferometer = simulator.via_plane_from(plane=plane, grid=grid)

    interferometer = interferometer.apply_settings(
        settings=ag.SettingsInterferometer(transformer_class=ag.TransformerDFT)
    )

    fit = ag.FitInterferometer(
        dataset=interferometer,
        plane=plane,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit.figure_of_merit == pytest.approx(-5.05513095, 1.0e-2)


def test__linear_light_profiles_agree_with_standard_light_profiles():

    grid = ag.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, sub_size=1)

    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.EllSersic(intensity=0.1, sersic_index=1.0),
        disk=ag.lp.EllSersic(intensity=0.2, sersic_index=4.0),
    )

    plane = ag.Plane(galaxies=[galaxy])

    simulator = ag.SimulatorInterferometer(
        uv_wavelengths=np.array(
            [
                [0.04, 200.0, 0.3, 400000.0, 60000000.0],
                [0.00003, 500.0, 600000.0, 0.1, 75555555],
            ]
        ),
        transformer_class=ag.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    interferometer = simulator.via_plane_from(plane=plane, grid=grid)

    interferometer = interferometer.apply_settings(
        settings=ag.SettingsInterferometer(
            grid_class=ag.Grid2D, transformer_class=ag.TransformerDFT, sub_size=1
        )
    )

    fit = ag.FitInterferometer(
        dataset=interferometer,
        plane=plane,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    galaxy_linear = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp_linear.EllSersic(sersic_index=1.0),
        disk=ag.lp_linear.EllSersic(sersic_index=4.0),
    )

    plane_linear = ag.Plane(galaxies=[galaxy_linear])

    fit_linear = ag.FitInterferometer(
        dataset=interferometer,
        plane=plane_linear,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
        settings_inversion=ag.SettingsInversion(
            use_w_tilde=False, linear_funcs_add_to_curvature_diag=False
        ),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array([0.1, 0.2]), 1.0e-2
    )
    assert fit_linear.linear_light_profile_intensity_dict[
        galaxy_linear.bulge
    ] == pytest.approx(0.1, 1.0e-2)
    assert fit_linear.linear_light_profile_intensity_dict[
        galaxy_linear.disk
    ] == pytest.approx(0.2, 1.0e-2)
    assert fit.log_likelihood == pytest.approx(fit_linear.log_likelihood, 1.0e-4)

    galaxy_image = galaxy.image_2d_from(grid=interferometer.grid)

    assert fit_linear.galaxy_model_image_dict[galaxy_linear] == pytest.approx(
        galaxy_image, 1.0e-4
    )

    galaxy_visibilities = galaxy.visibilities_from(
        grid=interferometer.grid, transformer=interferometer.transformer
    )

    assert fit_linear.galaxy_model_visibilities_dict[galaxy_linear] == pytest.approx(
        galaxy_visibilities, 1.0e-4
    )
