from astropy.io import fits

import os
from os import path
import numpy as np
import autogalaxy as ag


def create_fits(fits_path, array):

    if path.exists(fits_path):
        os.remove(fits_path)

    hdu_list = fits.HDUList()

    hdu_list.append(fits.ImageHDU(array))

    hdu_list.writeto(f"{fits_path}")


class TestImaging:
    def test__from_fits__all_imaging_data_structures_are_flipped_for_ds9(self):

        fits_path = path.join(
            "{}".format(path.dirname(path.realpath(__file__))), "files"
        )
        image_path = path.join(fits_path, "image.fits")

        create_fits(fits_path=image_path, array=[[1.0, 0.0], [0.0, 0.0]])

        noise_map_path = path.join(fits_path, "noise_map.fits")
        create_fits(fits_path=noise_map_path, array=[[2.0, 0.0], [0.0, 0.0]])

        psf_path = path.join(fits_path, "psf.fits")
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


class TestMaskedImaging:
    def test__masked_dataset_via_autoarray(self, imaging_7x7, sub_mask_7x7):

        masked_imaging_7x7 = ag.MaskedImaging(imaging=imaging_7x7, mask=sub_mask_7x7)

        assert (masked_imaging_7x7.image.slim == np.ones(9)).all()

        assert (
            masked_imaging_7x7.image.native == np.ones((7, 7)) * np.invert(sub_mask_7x7)
        ).all()

        assert (masked_imaging_7x7.noise_map.slim == 2.0 * np.ones(9)).all()
        assert (
            masked_imaging_7x7.noise_map.native
            == 2.0 * np.ones((7, 7)) * np.invert(sub_mask_7x7)
        ).all()

        assert (masked_imaging_7x7.psf.slim == (1.0 / 9.0) * np.ones(9)).all()
        assert (masked_imaging_7x7.psf.native == (1.0 / 9.0) * np.ones((3, 3))).all()

        assert type(masked_imaging_7x7.convolver) == ag.Convolver

    def test__inheritance_from_autoarray(
        self, imaging_7x7, sub_mask_7x7, blurring_grid_7x7
    ):

        masked_imaging_7x7 = ag.MaskedImaging(
            imaging=imaging_7x7,
            mask=sub_mask_7x7,
            settings=ag.SettingsMaskedImaging(
                grid_class=ag.Grid2D, psf_shape_2d=(3, 3)
            ),
        )

        grid = ag.Grid2D.from_mask(mask=sub_mask_7x7)

        assert (masked_imaging_7x7.grid == grid).all()

        blurring_grid = grid.blurring_grid_from_kernel_shape(kernel_shape_native=(3, 3))

        assert (masked_imaging_7x7.blurring_grid.slim == blurring_grid_7x7).all()
        assert (masked_imaging_7x7.blurring_grid == blurring_grid).all()

    def test__modified_image_and_noise_map(
        self, image_7x7, noise_map_7x7, imaging_7x7, sub_mask_7x7
    ):

        masked_imaging_7x7 = ag.MaskedImaging(imaging=imaging_7x7, mask=sub_mask_7x7)

        image_7x7[0] = 10.0
        noise_map_7x7[0] = 11.0

        masked_imaging_7x7 = masked_imaging_7x7.modify_image_and_noise_map(
            image=image_7x7, noise_map=noise_map_7x7
        )

        assert masked_imaging_7x7.image.slim[0] == 10.0
        assert masked_imaging_7x7.image.native[0, 0] == 10.0
        assert masked_imaging_7x7.noise_map.slim[0] == 11.0
        assert masked_imaging_7x7.noise_map.native[0, 0] == 11.0


class TestSimulatorImaging:
    def test__from_plane_and_grid__same_as_plane_image(self):

        psf = ag.Kernel2D.from_gaussian(
            shape_native=(7, 7), sigma=0.5, pixel_scales=0.05
        )

        grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

        galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0))

        galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.EllipticalSersic(intensity=0.3))

        plane = ag.Plane(redshift=0.75, galaxies=[galaxy_0, galaxy_1])

        simulator = ag.SimulatorImaging(
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_poisson_noise=False,
        )

        imaging = simulator.from_plane_and_grid(plane=plane, grid=grid)

        imaging_via_image = simulator.from_image(image=plane.image_from_grid(grid=grid))

        assert imaging.shape_native == (20, 20)
        assert imaging.image.native[0, 0] != imaging_via_image.image.native[0, 0]
        assert imaging.image.native[10, 10] == imaging_via_image.image.native[10, 10]
        assert (imaging.psf == imaging_via_image.psf).all()
        assert (imaging.noise_map == imaging_via_image.noise_map).all()

    def test__simulate_imaging_from_galaxy__source_galaxy__compare_to_imaging(self):

        galaxy_0 = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
            ),
        )

        galaxy_1 = ag.Galaxy(
            redshift=1.0,
            light=ag.lp.EllipticalSersic(
                centre=(0.1, 0.1),
                elliptical_comps=(0.096225, -0.055555),
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2, sub_size=1)

        psf = ag.Kernel2D.no_blur(pixel_scales=0.2)

        simulator = ag.SimulatorImaging(
            psf=psf,
            exposure_time=10000.0,
            background_sky_level=100.0,
            add_poisson_noise=True,
            noise_seed=1,
        )

        imaging = simulator.from_galaxies_and_grid(
            galaxies=[galaxy_0, galaxy_1], grid=grid
        )

        plane = ag.Plane(redshift=0.75, galaxies=[galaxy_0, galaxy_1])

        imaging_via_image = simulator.from_image(image=plane.image_from_grid(grid=grid))

        assert imaging.shape_native == (11, 11)
        assert (imaging.image == imaging_via_image.image).all()
        assert (imaging.psf == imaging_via_image.psf).all()
        assert imaging.noise_map == imaging_via_image.noise_map
