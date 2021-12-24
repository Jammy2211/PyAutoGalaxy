from astropy.io import fits
import numpy as np
import os
from os import path
import pytest
import shutil

import autogalaxy as ag


def create_fits(fits_path, array):

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


class TestImaging:
    def test__from_fits__all_imaging_data_structures_are_flipped_for_ds9(self):

        fits_path = path.join(
            "{}".format(path.dirname(path.realpath(__file__))), "files"
        )
        image_path = path.join(fits_path, "image.fits")
        noise_map_path = path.join(fits_path, "noise_map.fits")
        psf_path = path.join(fits_path, "psf.fits")

        create_fits(fits_path=image_path, array=[[1.0, 0.0], [0.0, 0.0]])
        create_fits(fits_path=noise_map_path, array=[[2.0, 0.0], [0.0, 0.0]])
        create_fits(fits_path=psf_path, array=[[1.0, 1.0], [0.0, 0.0]])

        imaging = ag.Imaging.from_fits(
            image_path=image_path,
            noise_map_path=noise_map_path,
            psf_path=psf_path,
            pixel_scales=0.1,
        )

        assert (imaging.image.native == np.array([[0.0, 0.0], [1.0, 0.0]])).all()
        assert (imaging.noise_map.native == np.array([[0.0, 0.0], [2.0, 0.0]])).all()
        assert (imaging.psf.native == np.array([[0.0, 0.0], [0.5, 0.5]])).all()

        imaging.output_to_fits(
            image_path=image_path,
            noise_map_path=noise_map_path,
            psf_path=psf_path,
            overwrite=True,
        )

        hdu_list = fits.open(image_path)
        image = np.array(hdu_list[0].data).astype("float64")
        assert (image == np.array([[1.0, 0.0], [0.0, 0.0]])).all()

        hdu_list = fits.open(noise_map_path)
        noise_map = np.array(hdu_list[0].data).astype("float64")
        assert (noise_map == np.array([[2.0, 0.0], [0.0, 0.0]])).all()

        hdu_list = fits.open(psf_path)
        psf = np.array(hdu_list[0].data).astype("float64")
        assert (psf == np.array([[0.5, 0.5], [0.0, 0.0]])).all()

        clean_fits(fits_path=fits_path)


class TestSimulatorImaging:
    def test__via_plane_from__same_as_plane_image(self):

        psf = ag.Kernel2D.from_gaussian(
            shape_native=(7, 7), sigma=0.5, pixel_scales=0.05
        )

        grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

        galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=1.0))

        galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.EllSersic(intensity=0.3))

        plane = ag.Plane(redshift=0.75, galaxies=[galaxy_0, galaxy_1])

        simulator = ag.SimulatorImaging(
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_poisson_noise=False,
        )

        imaging = simulator.via_plane_from(plane=plane, grid=grid)

        imaging_via_image = simulator.via_image_from(
            image=plane.image_2d_from(grid=grid)
        )

        assert imaging.shape_native == (20, 20)
        assert imaging.image.native[0, 0] != imaging_via_image.image.native[0, 0]
        assert imaging.image.native[10, 10] == imaging_via_image.image.native[10, 10]
        assert (imaging.psf == imaging_via_image.psf).all()
        assert (imaging.noise_map == imaging_via_image.noise_map).all()

    def test__simulate_imaging_from_galaxy__source_galaxy__compare_to_imaging(self):

        galaxy_0 = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
            ),
        )

        galaxy_1 = ag.Galaxy(
            redshift=1.0,
            light=ag.lp.EllSersic(
                centre=(0.1, 0.1),
                elliptical_comps=(0.096225, -0.055555),
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, sub_size=1)

        psf = ag.Kernel2D.manual_native(array=[[1.0]], pixel_scales=0.2)

        simulator = ag.SimulatorImaging(
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_poisson_noise=True,
            noise_seed=1,
        )

        imaging = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

        plane = ag.Plane(redshift=0.75, galaxies=[galaxy_0, galaxy_1])

        imaging_via_image = simulator.via_image_from(
            image=plane.image_2d_from(grid=grid)
        )

        assert imaging.shape_native == (11, 11)
        assert imaging.image == pytest.approx(imaging_via_image.image, 1.0e-4)
        assert (imaging.psf == imaging_via_image.psf).all()
        assert (imaging.noise_map == imaging_via_image.noise_map).all()
