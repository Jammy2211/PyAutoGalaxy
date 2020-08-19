import os
import shutil

import autogalaxy as ag
import numpy as np
import pytest


def test__simulate_imaging_data_and_fit__no_psf_blurring__chi_squared_is_0__noise_normalization_correct():

    grid = ag.GridIterate.uniform(shape_2d=(11, 11), pixel_scales=0.2)

    psf = ag.Kernel.manual_2d(
        array=np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        pixel_scales=0.2,
    )

    galaxy_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=ag.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )

    source_galaxy = ag.Galaxy(
        redshift=1.0,
        light=ag.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )

    plane = ag.Plane(galaxies=[galaxy_galaxy, source_galaxy])

    simulator = ag.SimulatorImaging(
        exposure_time_map=ag.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
        psf=psf,
        background_sky_map=ag.Array.zeros(shape_2d=grid.shape_2d),
        add_noise=False,
    )

    imaging = simulator.from_plane_and_grid(plane=plane, grid=grid)

    imaging.noise_map = ag.Array.ones(shape_2d=imaging.image.shape_2d)

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    imaging.output_to_fits(
        image_path=f"{path}/image.fits",
        noise_map_path=f"{path}/noise_map.fits",
        psf_path=f"{path}/psf.fits",
    )

    imaging = ag.Imaging.from_fits(
        image_path=f"{path}/image.fits",
        noise_map_path=f"{path}/noise_map.fits",
        psf_path=f"{path}/psf.fits",
        pixel_scales=0.2,
    )

    mask = ag.Mask.circular(
        shape_2d=imaging.image.shape_2d, pixel_scales=0.2, sub_size=2, radius=0.8
    )

    masked_imaging = ag.MaskedImaging(
        imaging=imaging,
        mask=mask,
        settings=ag.SettingsMaskedImaging(grid_class=ag.GridIterate),
    )

    plane = ag.Plane(galaxies=[galaxy_galaxy, source_galaxy])

    fit = ag.FitImaging(masked_imaging=masked_imaging, plane=plane)

    assert fit.chi_squared == 0.0

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)


def test__simulate_imaging_data_and_fit__include_psf_blurring__chi_squared_is_0__noise_normalization_correct():

    grid = ag.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.2, sub_size=1)

    psf = ag.Kernel.from_gaussian(
        shape_2d=(3, 3), pixel_scales=0.2, sigma=0.75, renormalize=True
    )

    galaxy_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=ag.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.8),
    )
    source_galaxy = ag.Galaxy(
        redshift=1.0,
        light=ag.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )
    plane = ag.Plane(galaxies=[galaxy_galaxy, source_galaxy])

    simulator = ag.SimulatorImaging(
        exposure_time_map=ag.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
        psf=psf,
        background_sky_map=ag.Array.zeros(shape_2d=grid.shape_2d),
        add_noise=False,
    )

    imaging = simulator.from_plane_and_grid(plane=plane, grid=grid)
    imaging.noise_map = ag.Array.ones(shape_2d=imaging.image.shape_2d)

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    imaging.output_to_fits(
        image_path=f"{path}/image.fits",
        noise_map_path=f"{path}/noise_map.fits",
        psf_path=f"{path}/psf.fits",
    )

    simulator = ag.Imaging.from_fits(
        image_path=f"{path}/image.fits",
        noise_map_path=f"{path}/noise_map.fits",
        psf_path=f"{path}/psf.fits",
        pixel_scales=0.2,
    )

    mask = ag.Mask.circular(
        shape_2d=simulator.image.shape_2d, pixel_scales=0.2, sub_size=1, radius=0.8
    )

    masked_imaging = ag.MaskedImaging(
        imaging=simulator,
        mask=mask,
        settings=ag.SettingsMaskedImaging(grid_class=ag.Grid, sub_size=1),
    )

    plane = ag.Plane(galaxies=[galaxy_galaxy, source_galaxy])

    fit = ag.FitImaging(masked_imaging=masked_imaging, plane=plane)

    assert fit.chi_squared == pytest.approx(0.0, 1e-4)

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)


def test__simulate_interferometer_data_and_fit__chi_squared_is_0__noise_normalization_correct():

    grid = ag.Grid.uniform(shape_2d=(51, 51), pixel_scales=0.1, sub_size=2)

    galaxy_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=ag.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = ag.Galaxy(
        redshift=1.0,
        light=ag.lp.EllipticalExponential(centre=(0.1, 0.1), intensity=0.5),
    )

    plane = ag.Plane(galaxies=[galaxy_galaxy, source_galaxy])

    simulator = ag.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        transformer_class=ag.TransformerDFT,
        exposure_time_map=ag.Array.full(fill_value=300.0, shape_2d=grid.shape_2d),
        background_sky_map=ag.Array.zeros(shape_2d=grid.shape_2d),
        noise_if_add_noise_false=1.0,
        noise_sigma=None,
    )

    interferometer = simulator.from_plane_and_grid(plane=plane, grid=grid)

    path = "{}/data_temp/simulate_and_fit".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    try:
        shutil.rmtree(path)
    except FileNotFoundError:
        pass

    if os.path.exists(path) is False:
        os.makedirs(path)

    interferometer.output_to_fits(
        visibilities_path=path + "/visibilities.fits",
        noise_map_path=f"{path}/noise_map.fits",
        uv_wavelengths_path=path + "/uv_wavelengths.fits",
    )

    interferometer = ag.Interferometer.from_fits(
        visibilities_path=path + "/visibilities.fits",
        noise_map_path=f"{path}/noise_map.fits",
        uv_wavelengths_path=path + "/uv_wavelengths.fits",
    )

    visibilities_mask = np.full(fill_value=False, shape=(7, 2))

    real_space_mask = ag.Mask.unmasked(shape_2d=(51, 51), pixel_scales=0.1, sub_size=2)

    masked_interferometer = ag.MaskedInterferometer(
        interferometer=interferometer,
        visibilities_mask=visibilities_mask,
        real_space_mask=real_space_mask,
        settings=ag.SettingsMaskedInterferometer(
            grid_class=ag.Grid, transformer_class=ag.TransformerDFT, sub_size=2
        ),
    )

    plane = ag.Plane(galaxies=[galaxy_galaxy, source_galaxy])

    fit = ag.FitInterferometer(
        masked_interferometer=masked_interferometer,
        plane=plane,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
    )

    assert fit.chi_squared == pytest.approx(0.0)

    pix = ag.pix.Rectangular(shape=(7, 7))

    reg = ag.reg.Constant(coefficient=0.0001)

    galaxy_galaxy = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.EllipticalSersic(centre=(0.1, 0.1), intensity=0.1),
        mass=ag.mp.EllipticalIsothermal(centre=(0.1, 0.1), einstein_radius=1.0),
    )

    source_galaxy = ag.Galaxy(redshift=1.0, pixelization=pix, regularization=reg)

    plane = ag.Plane(galaxies=[galaxy_galaxy, source_galaxy])

    fit = ag.FitInterferometer(
        masked_interferometer=masked_interferometer,
        plane=plane,
        settings_pixelization=ag.SettingsPixelization(use_border=False),
    )
    assert abs(fit.chi_squared) < 1.0e-4

    path = "{}/data_temp".format(
        os.path.dirname(os.path.realpath(__file__))
    )  # Setup path so we can output the simulated image.

    if os.path.exists(path) == True:
        shutil.rmtree(path)
