import os
from os import path
import shutil

import numpy as np
import pytest

import autogalaxy as ag


def test__perfect_fit__chi_squared_0():
    grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, sub_size=1)

    psf = ag.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy_0 = ag.Galaxy(
        redshift=0.5, light=ag.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1)
    )
    galaxy_1 = ag.Galaxy(
        redshift=0.5, light=ag.lp.EllExponential(centre=(0.1, 0.1), intensity=0.5)
    )
    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise=False
    )

    imaging = simulator.via_plane_from(plane=plane, grid=grid)
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
        sub_size=1,
        radius=0.8,
    )

    masked_imaging = imaging.apply_mask(mask=mask)
    masked_imaging = masked_imaging.apply_settings(
        settings=ag.SettingsImaging(grid_class=ag.Grid2D, sub_size=1)
    )

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    fit = ag.FitImaging(dataset=masked_imaging, plane=plane)

    assert fit.chi_squared == pytest.approx(0.0, 1e-4)

    file_path = path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "data_temp"
    )

    if path.exists(file_path) is True:
        shutil.rmtree(file_path)


def test__simulate_imaging_data_and_fit__known_likelihood():

    grid = ag.Grid2D.uniform(shape_native=(31, 31), pixel_scales=0.2, sub_size=1)

    psf = ag.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy_0 = ag.Galaxy(
        redshift=0.5, light=ag.lp.EllSersic(centre=(0.1, 0.1), intensity=0.1)
    )
    galaxy_1 = ag.Galaxy(
        redshift=1.0,
        pixelization=ag.pix.Rectangular(shape=(16, 16)),
        regularization=ag.reg.Constant(coefficient=(1.0)),
    )

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    imaging = ag.SimulatorImaging(exposure_time=300.0, psf=psf, noise_seed=1)

    imaging = imaging.via_plane_from(plane=plane, grid=grid)

    mask = ag.Mask2D.circular(
        shape_native=imaging.image.shape_native, pixel_scales=0.2, radius=2.0
    )

    masked_imaging = imaging.apply_mask(mask=mask)

    fit = ag.FitImaging(dataset=masked_imaging, plane=plane)

    assert fit.figure_of_merit == pytest.approx(621.5594864161, 1.0e-2)


def test__simulate_imaging_data_and_fit__linear_light_profiles_agree_with_standard_light_profiles():

    grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, sub_size=1)

    psf = ag.Kernel2D.from_gaussian(
        shape_native=(3, 3), pixel_scales=0.2, sigma=0.75, normalize=True
    )

    galaxy = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp.EllSersic(intensity=0.1, sersic_index=1.0),
        disk=ag.lp.EllSersic(intensity=0.2, sersic_index=4.0),
    )

    plane = ag.Plane(galaxies=[galaxy])

    simulator = ag.SimulatorImaging(
        exposure_time=300.0, psf=psf, add_poisson_noise=False
    )

    imaging = simulator.via_plane_from(plane=plane, grid=grid)
    imaging.noise_map = ag.Array2D.ones(
        shape_native=imaging.image.shape_native, pixel_scales=0.2
    )

    mask = ag.Mask2D.circular(
        shape_native=imaging.image.shape_native,
        pixel_scales=0.2,
        sub_size=1,
        radius=0.8,
    )

    masked_imaging = imaging.apply_mask(mask=mask)
    masked_imaging = masked_imaging.apply_settings(
        settings=ag.SettingsImaging(grid_class=ag.Grid2D, sub_size=1)
    )

    plane = ag.Plane(galaxies=[galaxy])

    fit = ag.FitImaging(dataset=masked_imaging, plane=plane)

    galaxy_linear = ag.Galaxy(
        redshift=0.5,
        bulge=ag.lp_linear.EllSersic(sersic_index=1.0),
        disk=ag.lp_linear.EllSersic(sersic_index=4.0),
    )

    plane_linear = ag.Plane(galaxies=[galaxy_linear])

    fit_linear = ag.FitImaging(
        dataset=masked_imaging,
        plane=plane_linear,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )

    assert fit_linear.inversion.reconstruction == pytest.approx(
        np.array([0.1, 0.2]), 1.0e-4
    )

    assert fit_linear.linear_light_profile_intensity_dict[
        galaxy_linear.bulge
    ] == pytest.approx(0.1, 1.0e-4)
    assert fit_linear.linear_light_profile_intensity_dict[
        galaxy_linear.disk
    ] == pytest.approx(0.2, 1.0e-4)

    assert fit.log_likelihood == fit_linear.figure_of_merit
    assert fit_linear.figure_of_merit == pytest.approx(-45.02798, 1.0e-4)

    galaxy_image = galaxy.blurred_image_2d_from(
        grid=masked_imaging.grid,
        convolver=masked_imaging.convolver,
        blurring_grid=masked_imaging.blurring_grid,
    )

    assert fit_linear.galaxy_model_image_dict[galaxy_linear] == pytest.approx(
        galaxy_image, 1.0e-4
    )
