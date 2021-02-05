import os
from os import path
import shutil

import numpy as np
import pytest

import autogalaxy as ag


def test__simulate_imaging_data_and_fit__no_psf_blurring__chi_squared_is_0__noise_normalization_correct():
    grid = ag.Grid2DIterate.uniform(shape_native=(11, 11), pixel_scales=0.2)

    psf = ag.Kernel2D.manual_native(
        array=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], pixel_scales=0.2
    )

    lens_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=ag.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    source_galaxy = ag.Galaxy(
        redshift=1.0,
        light=ag.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )

    plane = ag.Plane(galaxies=[lens_galaxy, source_galaxy])

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise=False
    )

    imaging = simulator.from_plane_and_grid(plane=plane, grid=grid)

    imaging.noise_map = ag.Array2D.ones(
        shape_native=imaging.image.shape_native, pixel_scales=0.2
    )

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

    imaging.output_to_fits(
        image_path=path.join(file_path, "image.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
    )

    imaging = ag.Imaging.from_fits(
        image_path=path.join(file_path, "image.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        pixel_scales=0.2,
    )

    mask = ag.Mask2D.circular(
        shape_native=imaging.image.shape_native,
        pixel_scales=0.2,
        sub_size=2,
        radius=0.8,
    )

    masked_imaging = ag.MaskedImaging(
        imaging=imaging,
        mask=mask,
        settings=ag.SettingsMaskedImaging(grid_class=ag.Grid2DIterate),
    )

    plane = ag.Plane(galaxies=[lens_galaxy, source_galaxy])

    fit = ag.FitImaging(masked_imaging=masked_imaging, plane=plane)

    assert fit.chi_squared == 0.0

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) == True:
        shutil.rmtree(file_path)


def test__simulate_imaging_data_and_fit__include_psf_blurring__chi_squared_is_0__noise_normalization_correct():
    grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, sub_size=1)

    psf = ag.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, renormalize=True
    )

    lens_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=ag.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = ag.Galaxy(
        redshift=1.0,
        light=ag.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )
    plane = ag.Plane(galaxies=[lens_galaxy, source_galaxy])

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise=False
    )

    imaging = simulator.from_plane_and_grid(plane=plane, grid=grid)
    imaging.noise_map = ag.Array2D.ones(
        shape_native=imaging.image.shape_native, pixel_scales=0.2
    )

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

    imaging.output_to_fits(
        image_path=path.join(file_path, "image.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
    )

    simulator = ag.Imaging.from_fits(
        image_path=path.join(file_path, "image.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        psf_path=path.join(file_path, "psf.fits"),
        pixel_scales=0.2,
    )

    mask = ag.Mask2D.circular(
        shape_native=simulator.image.shape_native,
        pixel_scales=0.2,
        sub_size=1,
        radius=0.8,
    )

    masked_imaging = ag.MaskedImaging(
        imaging=simulator,
        mask=mask,
        settings=ag.SettingsMaskedImaging(grid_class=ag.Grid2D, sub_size=1),
    )

    plane = ag.Plane(galaxies=[lens_galaxy, source_galaxy])

    fit = ag.FitImaging(masked_imaging=masked_imaging, plane=plane)

    assert fit.chi_squared == pytest.approx(0.0, 1e-4)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) == True:
        shutil.rmtree(file_path)


def test__simulate_interferometer_data_and_fit__chi_squared_is_0__noise_normalization_correct():
    grid = ag.Grid2D.uniform(shape_native=(51, 51), pixel_scales=0.1, sub_size=2)

    lens_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=ag.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = ag.Galaxy(
        redshift=1.0,
        light=ag.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )

    plane = ag.Plane(galaxies=[lens_galaxy, source_galaxy])

    simulator = ag.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=ag.TransformerDFT,
        exposure_time=300.0,
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    interferometer = simulator.from_plane_and_grid(plane=plane, grid=grid)

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

    interferometer = ag.Interferometer.from_fits(
        visibilities_path=path.join(file_path, "visibilities.fits"),
        noise_map_path=path.join(file_path, "noise_map.fits"),
        uv_wavelengths_path=path.join(file_path, "uv_wavelengths.fits"),
    )

    visibilities_mask = np.full(fill_value=False, shape=(7,))

    real_space_mask = ag.Mask2D.unmasked(
        shape_native=(51, 51), pixel_scales=0.1, sub_size=2
    )

    masked_interferometer = ag.MaskedInterferometer(
        interferometer=interferometer,
        visibilities_mask=visibilities_mask,
        real_space_mask=real_space_mask,
        settings=ag.SettingsMaskedInterferometer(
            grid_class=ag.Grid2D, transformer_class=ag.TransformerDFT, sub_size=2
        ),
    )

    plane = ag.Plane(galaxies=[lens_galaxy, source_galaxy])

    fit = ag.FitInterferometer(
        masked_interferometer=masked_interferometer,
        plane=plane,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
    )

    assert fit.chi_squared == pytest.approx(0.0)

    pix = ag.pix.Rectangular(shape=(7, 7))

    reg = ag.reg.Constant(coefficient=0.0001)

    lens_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=ag.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(galaxies=[lens_galaxy, source_galaxy])

    fit = ag.FitInterferometer(
        masked_interferometer=masked_interferometer,
        plane=plane,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
    )
    assert abs(fit.chi_squared) < 1.0e-4

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    shutil.rmtree(file_path, ignore_errors=True)
