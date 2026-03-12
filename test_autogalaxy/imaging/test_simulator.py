import numpy as np
import os
from os import path
import pytest
import shutil

import autogalaxy as ag


def create_fits(fits_path, array):

    from astropy.io import fits

    file_dir = os.path.split(fits_path)[0]

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    if os.path.exists(fits_path):
        os.remove(fits_path)

    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU(array))

    hdu_list.writeto(f"{fits_path}")


def clean_fits(fits_path):
    if path.exists(fits_path):
        shutil.rmtree(fits_path)



def test__simulator__via_galaxies_from():
    psf = ag.Convolver.from_gaussian(shape_native=(7, 7), sigma=0.5, pixel_scales=0.05)

    grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=1.0))

    galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.Sersic(intensity=0.3))

    simulator = ag.SimulatorImaging(
        psf=psf,
        exposure_time=10000.0,
        background_sky_level=100.0,
        add_poisson_noise_to_data=False,
        include_poisson_noise_in_noise_map=False,
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

    imaging_via_image = simulator.via_image_from(
        image=galaxies.image_2d_from(grid=grid)
    )

    assert dataset.shape_native == (20, 20)
    assert dataset.data.native[0, 0] != imaging_via_image.data.native[0, 0]
    assert dataset.data.native[10, 10] == pytest.approx(
        imaging_via_image.data.native[10, 10], 1.0e-4
    )
    assert dataset.psf.kernel == pytest.approx(imaging_via_image.psf.kernel, 1.0e-4)
    assert dataset.noise_map == pytest.approx(imaging_via_image.noise_map, 1.0e-4)


def test__simulator__simulate_imaging_from_galaxy__source_galaxy__compare_to_imaging():
    galaxy_0 = ag.Galaxy(
        redshift=0.5,
        mass=ag.mp.Isothermal(
            centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
        ),
    )

    galaxy_1 = ag.Galaxy(
        redshift=1.0,
        light=ag.lp.Sersic(
            centre=(0.1, 0.1),
            ell_comps=(0.096225, -0.055555),
            intensity=0.3,
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )

    grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2)

    kernel = ag.Array2D.no_mask(values=[[1.0]], pixel_scales=0.2)
    psf = ag.Convolver(kernel=kernel)

    simulator = ag.SimulatorImaging(
        psf=psf,
        exposure_time=10000.0,
        background_sky_level=100.0,
        add_poisson_noise_to_data=True,
        noise_seed=1,
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

    imaging_via_image = simulator.via_image_from(
        image=galaxies.image_2d_from(grid=grid)
    )

    assert dataset.shape_native == (11, 11)
    assert dataset.data.array == pytest.approx(imaging_via_image.data.array, 1.0e-4)
    assert (dataset.psf.kernel == imaging_via_image.psf.kernel).all()
    assert dataset.noise_map == pytest.approx(imaging_via_image.noise_map, 1.0e-4)
