from __future__ import division, print_function

import math

import numpy as np
import pytest
import scipy.special

import autogalaxy as ag
from autogalaxy.mock import mock

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestPointSources:
    def test__simple(self):

        point_source = ag.lp.PointSource(centre=(0.0, 0.0))

        assert point_source.centre == (0.0, 0.0)

        point_source = ag.lp.PointSourceFlux(centre=(0.0, 0.0), flux=0.1)

        assert point_source.centre == (0.0, 0.0)
        assert point_source.flux == 0.1


class TestGaussian:
    def test__intensity_as_radius__correct_value(self):
        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
        )

        image = gaussian.image_from_grid_radii(grid_radii=1.0)

        assert image == pytest.approx(0.60653, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=2.0, sigma=1.0
        )

        image = gaussian.image_from_grid_radii(grid_radii=1.0)

        assert image == pytest.approx(2.0 * 0.60653, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
        )

        image = gaussian.image_from_grid_radii(grid_radii=1.0)

        assert image == pytest.approx(0.882496, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
        )

        image = gaussian.image_from_grid_radii(grid_radii=3.0)

        assert image == pytest.approx(0.32465, 1e-2)

    def test__image_from_grid__same_values_as_above(self):
        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
        )

        image = gaussian.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(0.60653, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=2.0, sigma=1.0
        )

        image = gaussian.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(2.0 * 0.60653, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
        )

        image = gaussian.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(0.88249, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
        )

        image = gaussian.image_from_grid(grid=np.array([[0.0, 3.0]]))

        assert image == pytest.approx(0.3246, 1e-2)

    def test__image_from_grid__change_geometry(self):
        gaussian = ag.lp.EllipticalGaussian(
            centre=(1.0, 1.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
        )
        image = gaussian.image_from_grid(grid=np.array([[1.0, 0.0]]))
        assert image == pytest.approx(0.60653, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.333333),
            intensity=1.0,
            sigma=1.0,
        )

        image = gaussian.image_from_grid(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(0.60653, 1e-2)

        gaussian_0 = ag.lp.EllipticalGaussian(
            centre=(-3.0, -0.0),
            elliptical_comps=(0.0, 0.333333),
            intensity=1.0,
            sigma=1.0,
        )

        gaussian_1 = ag.lp.EllipticalGaussian(
            centre=(3.0, 0.0),
            elliptical_comps=(0.0, 0.333333),
            intensity=1.0,
            sigma=1.0,
        )

        image_0 = gaussian_0.image_from_grid(
            grid=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        )

        image_1 = gaussian_1.image_from_grid(
            grid=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        )

        assert image_0 == pytest.approx(image_1, 1e-4)

        gaussian_0 = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.333333),
            intensity=1.0,
            sigma=1.0,
        )

        gaussian_1 = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.333333),
            intensity=1.0,
            sigma=1.0,
        )

        image_0 = gaussian_0.image_from_grid(
            grid=np.array([[0.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        )

        image_1 = gaussian_1.image_from_grid(
            grid=np.array([[0.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
        )

        assert image_0 == pytest.approx(image_1, 1e-4)

    def test__spherical_and_elliptical_match(self):
        elliptical = ag.lp.EllipticalGaussian(
            elliptical_comps=(0.0, 0.0), intensity=3.0, sigma=2.0
        )
        spherical = ag.lp.SphericalGaussian(intensity=3.0, sigma=2.0)

        image_elliptical = elliptical.image_from_grid(grid=grid)
        image_spherical = spherical.image_from_grid(grid=grid)

        assert (image_elliptical == image_spherical).all()

    def test__output_image_is_array(self):
        grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)

        gaussian = ag.lp.EllipticalGaussian()

        image = gaussian.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)

        gaussian = ag.lp.SphericalGaussian()

        image = gaussian.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)


class TestSersic:
    def test__image_from_grid_radii__correct_value(self):
        sersic = ag.lp.EllipticalSersic(
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
        )

        image = sersic.image_from_grid_radii(grid_radii=1.0)

        assert image == pytest.approx(0.351797, 1e-3)

        sersic = ag.lp.EllipticalSersic(
            elliptical_comps=(0.0, 0.0),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )
        # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797

        image = sersic.image_from_grid_radii(grid_radii=1.5)

        assert image == pytest.approx(4.90657319276, 1e-3)

    def test__image_from_grid__correct_values_for_input_parameters(self):
        sersic = ag.lp.EllipticalSersic(
            elliptical_comps=(0.0, 0.333333),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )

        image = sersic.image_from_grid(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(5.38066670129, 1e-3)

    def test__image_from_grid__change_geometry(self):
        sersic_0 = ag.lp.EllipticalSersic(
            elliptical_comps=(0.0, 0.333333),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )

        sersic_1 = ag.lp.EllipticalSersic(
            elliptical_comps=(0.0, -0.333333),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )

        image_0 = sersic_0.image_from_grid(grid=np.array([[0.0, 1.0]]))

        image_1 = sersic_1.image_from_grid(grid=np.array([[1.0, 0.0]]))

        assert (image_0 == image_1).all()

    def test__spherical_and_elliptical_match(self):
        elliptical = ag.lp.EllipticalSersic(
            elliptical_comps=(0.0, 0.0),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )

        spherical = ag.lp.SphericalSersic(
            intensity=3.0, effective_radius=2.0, sersic_index=2.0
        )

        image_elliptical = elliptical.image_from_grid(grid=grid)

        image_spherical = spherical.image_from_grid(grid=grid)

        assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)

    def test__output_image_is_autoarray(self):
        grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)

        sersic = ag.lp.EllipticalSersic()

        image = sersic.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)

        sersic = ag.lp.SphericalSersic()

        image = sersic.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)


class TestExponential:
    def test__image_from_grid_radii__correct_value(self):
        exponential = ag.lp.EllipticalExponential(
            elliptical_comps=(0.0, 0.0), intensity=1.0, effective_radius=0.6
        )

        image = exponential.image_from_grid_radii(grid_radii=1.0)

        assert image == pytest.approx(0.3266, 1e-3)

        exponential = ag.lp.EllipticalExponential(
            elliptical_comps=(0.0, 0.0), intensity=3.0, effective_radius=2.0
        )

        image = exponential.image_from_grid_radii(grid_radii=1.5)

        assert image == pytest.approx(4.5640, 1e-3)

    def test__image_from_grid__correct_values(self):
        exponential = ag.lp.EllipticalExponential(
            elliptical_comps=(0.0, 0.333333), intensity=3.0, effective_radius=2.0
        )

        image = exponential.image_from_grid(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(4.9047, 1e-3)

        exponential = ag.lp.EllipticalExponential(
            elliptical_comps=(0.0, -0.333333), intensity=2.0, effective_radius=3.0
        )

        image = exponential.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(4.8566, 1e-3)

        exponential = ag.lp.EllipticalExponential(
            elliptical_comps=(0.0, -0.333333), intensity=4.0, effective_radius=3.0
        )

        image = exponential.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(2.0 * 4.8566, 1e-3)

        value = exponential.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert value == pytest.approx(2.0 * 4.8566, 1e-3)

    def test__image_from_grid__change_geometry(self):
        exponential_0 = ag.lp.EllipticalExponential(
            elliptical_comps=(0.0, 0.333333), intensity=3.0, effective_radius=2.0
        )

        exponential_1 = ag.lp.EllipticalExponential(
            elliptical_comps=(0.0, -0.333333), intensity=3.0, effective_radius=2.0
        )

        image_0 = exponential_0.image_from_grid(grid=np.array([[0.0, 1.0]]))

        image_1 = exponential_1.image_from_grid(grid=np.array([[1.0, 0.0]]))

        assert (image_0 == image_1).all()

    def test__spherical_and_elliptical_match(self):
        elliptical = ag.lp.EllipticalExponential(
            elliptical_comps=(0.0, 0.0), intensity=3.0, effective_radius=2.0
        )

        spherical = ag.lp.SphericalExponential(intensity=3.0, effective_radius=2.0)

        image_elliptical = elliptical.image_from_grid(grid=grid)
        image_spherical = spherical.image_from_grid(grid=grid)

        assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)

    def test__output_image_is_autoarray(self):
        grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)

        exponential = ag.lp.EllipticalExponential()

        image = exponential.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)

        exponential = ag.lp.SphericalExponential()

        image = exponential.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)


class TestDevVaucouleurs:
    def test__image_from_grid_radii__correct_value(self):
        dev_vaucouleurs = ag.lp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.0, 0.0), intensity=1.0, effective_radius=0.6
        )

        image = dev_vaucouleurs.image_from_grid_radii(grid_radii=1.0)

        assert image == pytest.approx(0.3518, 1e-3)

        dev_vaucouleurs = ag.lp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.0, 0.0), intensity=3.0, effective_radius=2.0
        )

        image = dev_vaucouleurs.image_from_grid_radii(grid_radii=1.5)

        assert image == pytest.approx(5.1081, 1e-3)

    def test__image_from_grid__correct_values(self):
        dev_vaucouleurs = ag.lp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.0, 0.333333), intensity=3.0, effective_radius=2.0
        )

        image = dev_vaucouleurs.image_from_grid(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(5.6697, 1e-3)

        dev_vaucouleurs = ag.lp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.0, -0.333333), intensity=2.0, effective_radius=3.0
        )

        image = dev_vaucouleurs.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(7.4455, 1e-3)

        dev_vaucouleurs = ag.lp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.0, -0.333333), intensity=4.0, effective_radius=3.0
        )

        image = dev_vaucouleurs.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(2.0 * 7.4455, 1e-3)

        value = dev_vaucouleurs.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert value == pytest.approx(2.0 * 7.4455, 1e-3)

    def test__image_from_grid__change_geometry(self):
        dev_vaucouleurs_0 = ag.lp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.0, 0.333333), intensity=3.0, effective_radius=2.0
        )

        dev_vaucouleurs_1 = ag.lp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.0, -0.333333), intensity=3.0, effective_radius=2.0
        )

        image_0 = dev_vaucouleurs_0.image_from_grid(grid=np.array([[0.0, 1.0]]))

        image_1 = dev_vaucouleurs_1.image_from_grid(grid=np.array([[1.0, 0.0]]))

        assert image_0 == image_1

    def test__spherical_and_elliptical_match(self):
        elliptical = ag.lp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.0, 0.0), intensity=3.0, effective_radius=2.0
        )

        spherical = ag.lp.SphericalDevVaucouleurs(intensity=3.0, effective_radius=2.0)

        image_elliptical = elliptical.image_from_grid(grid=grid)

        image_spherical = spherical.image_from_grid(grid=grid)

        assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)

    def test__output_image_is_autoarray(self):
        grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)

        dev_vaucouleurs = ag.lp.EllipticalDevVaucouleurs()

        image = dev_vaucouleurs.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)

        dev_vaucouleurs = ag.lp.SphericalDevVaucouleurs()

        image = dev_vaucouleurs.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)


class TestCoreSersic:
    def test__image_from_grid_radii__correct_value(self):
        core_sersic = ag.lp.EllipticalCoreSersic(
            elliptical_comps=(0.0, 0.333333),
            intensity=1.0,
            effective_radius=5.0,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=1.0,
        )

        image = core_sersic.image_from_grid_radii(0.01)

        assert image == 0.1

    def test__spherical_and_elliptical_match(self):
        elliptical = ag.lp.EllipticalCoreSersic(
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=5.0,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=1.0,
        )

        spherical = ag.lp.SphericalCoreSersic(
            intensity=1.0,
            effective_radius=5.0,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=1.0,
        )

        image_elliptical = elliptical.image_from_grid(grid=grid)

        image_spherical = spherical.image_from_grid(grid=grid)

        assert (image_elliptical == image_spherical).all()

    def test__output_image_is_autoarray(self):
        grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)

        core_sersic = ag.lp.EllipticalCoreSersic()

        image = core_sersic.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)

        core_sersic = ag.lp.SphericalCoreSersic()

        image = core_sersic.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)


class TestChameleon:
    def test__image_from_grid_radii__correct_value(self):
        chameleon = ag.lp.EllipticalChameleon(
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
        )

        image = chameleon.image_from_grid_radii(grid_radii=1.0)

        assert image == pytest.approx(0.018605, 1e-3)

        chameleon = ag.lp.EllipticalChameleon(
            elliptical_comps=(0.5, 0.0),
            intensity=3.0,
            core_radius_0=0.2,
            core_radius_1=0.4,
        )
        # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797

        image = chameleon.image_from_grid_radii(grid_radii=1.5)

        assert image == pytest.approx(0.07816, 1e-3)

    def test__image_from_grid__correct_values_for_input_parameters(self):
        chameleon = ag.lp.EllipticalChameleon(
            elliptical_comps=(0.0, 0.333333),
            intensity=3.0,
            core_radius_0=0.2,
            core_radius_1=0.4,
        )

        image = chameleon.image_from_grid(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(0.024993, 1e-3)

    def test__image_from_grid__change_geometry(self):
        chameleon_0 = ag.lp.EllipticalChameleon(
            elliptical_comps=(0.0, 0.333333),
            intensity=3.0,
            core_radius_0=0.2,
            core_radius_1=0.4,
        )

        chameleon_1 = ag.lp.EllipticalChameleon(
            elliptical_comps=(0.0, -0.333333),
            intensity=3.0,
            core_radius_0=0.2,
            core_radius_1=0.4,
        )

        image_0 = chameleon_0.image_from_grid(grid=np.array([[0.0, 1.0]]))

        image_1 = chameleon_1.image_from_grid(grid=np.array([[1.0, 0.0]]))

        assert (image_0 == image_1).all()

    def _test__spherical_and_elliptical_match(self):
        elliptical = ag.lp.EllipticalChameleon(
            elliptical_comps=(0.0, 0.0),
            intensity=3.0,
            core_radius_0=0.2,
            core_radius_1=0.4,
        )

        spherical = ag.lp.SphericalChameleon(
            intensity=3.0, core_radius_0=0.2, core_radius_1=0.4
        )

        image_elliptical = elliptical.image_from_grid(grid=grid)

        image_spherical = spherical.image_from_grid(grid=grid)

        assert (image_elliptical == image_spherical).all()

    def test__output_image_is_autoarray(self):
        grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)

        chameleon = ag.lp.EllipticalChameleon()

        image = chameleon.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)

        chameleon = ag.lp.SphericalChameleon()

        image = chameleon.image_from_grid(grid=grid)

        assert image.shape_native == (2, 2)


class TestBlurredProfileImages:
    def test__blurred_image_from_grid_and_psf(
        self, sub_grid_7x7, blurring_grid_7x7, psf_3x3, convolver_7x7
    ):
        light_profile = ag.lp.EllipticalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=sub_grid_7x7)

        blurring_image = light_profile.image_from_grid(grid=blurring_grid_7x7)

        blurred_image = convolver_7x7.convolved_image_from_image_and_blurring_image(
            image=image.slim_binned, blurring_image=blurring_image.slim_binned
        )

        light_profile_blurred_image = light_profile.blurred_image_from_grid_and_psf(
            grid=sub_grid_7x7, blurring_grid=blurring_grid_7x7, psf=psf_3x3
        )

        assert blurred_image.slim == pytest.approx(
            light_profile_blurred_image.slim, 1.0e-4
        )
        assert blurred_image.native == pytest.approx(
            light_profile_blurred_image.native, 1.0e-4
        )

    def test__blurred_image_from_grid_and_convolver(
        self, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
    ):
        light_profile = ag.lp.EllipticalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=sub_grid_7x7)

        blurring_image = light_profile.image_from_grid(grid=blurring_grid_7x7)

        blurred_image = convolver_7x7.convolved_image_from_image_and_blurring_image(
            image=image.slim_binned, blurring_image=blurring_image.slim_binned
        )

        light_profile_blurred_image = light_profile.blurred_image_from_grid_and_convolver(
            grid=sub_grid_7x7, convolver=convolver_7x7, blurring_grid=blurring_grid_7x7
        )

        assert blurred_image.slim == pytest.approx(
            light_profile_blurred_image.slim, 1.0e-4
        )
        assert blurred_image.native == pytest.approx(
            light_profile_blurred_image.native, 1.0e-4
        )


class TestVisibilities:
    def test__visibilities_from_grid_and_transformer(
        self, grid_7x7, sub_grid_7x7, transformer_7x7_7
    ):
        light_profile = ag.lp.EllipticalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=grid_7x7)

        visibilities = transformer_7x7_7.visibilities_from_image(
            image=image.slim_binned
        )

        light_profile_visibilities = light_profile.profile_visibilities_from_grid_and_transformer(
            grid=grid_7x7, transformer=transformer_7x7_7
        )

        assert visibilities == pytest.approx(light_profile_visibilities, 1.0e-4)


def luminosity_from_radius_and_profile(radius, profile):
    x = profile.sersic_constant * (
        (radius / profile.effective_radius) ** (1.0 / profile.sersic_index)
    )

    return (
        profile.intensity
        * profile.effective_radius ** 2
        * 2
        * math.pi
        * profile.sersic_index
        * (
            (math.e ** profile.sersic_constant)
            / (profile.sersic_constant ** (2 * profile.sersic_index))
        )
        * scipy.special.gamma(2 * profile.sersic_index)
        * scipy.special.gammainc(2 * profile.sersic_index, x)
    )


class TestLuminosityWithinCircle:
    def test__compare_to_analytic_and_gridded_luminosity_calculations(self):

        sersic = ag.lp.SphericalSersic(
            intensity=3.0, effective_radius=2.0, sersic_index=2.0
        )

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=0.5, profile=sersic
        )

        luminosity_integral = sersic.luminosity_within_circle(radius=0.5)

        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

        luminosity_grid = luminosity_from_radius_and_profile(radius=1.0, profile=sersic)

        luminosity_integral = sersic.luminosity_within_circle(radius=1.0)

        assert luminosity_grid == pytest.approx(luminosity_integral, 0.02)


class TestDecorators:
    def test__grid_iterate_in__iterates_grid_correctly(self):
        mask = ag.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = ag.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        light_profile = ag.lp.EllipticalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = ag.Grid2D.from_mask(mask=mask_sub_2)
        image_sub_2 = light_profile.image_from_grid(grid=grid_sub_2).slim_binned

        assert (image == image_sub_2).all()

        grid = ag.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.95, sub_steps=[2, 4, 8]
        )

        light_profile = ag.lp.EllipticalSersic(centre=(0.08, 0.08), intensity=1.0)

        image = light_profile.image_from_grid(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = ag.Grid2D.from_mask(mask=mask_sub_4)
        image_sub_4 = light_profile.image_from_grid(grid=grid_sub_4).slim_binned

        assert image[0] == image_sub_4[0]

        mask_sub_8 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)
        grid_sub_8 = ag.Grid2D.from_mask(mask=mask_sub_8)
        image_sub_8 = light_profile.image_from_grid(grid=grid_sub_8).slim_binned

        assert image[4] == image_sub_8[4]

    def test__grid_iterate_in__iterates_grid_correctly_for_peak(self):
        grid = ag.Grid2DIterate.uniform(
            shape_native=(100, 100),
            pixel_scales=0.1,
            fractional_accuracy=0.9999,
            sub_steps=[2, 4, 8, 16, 24],
        )

        light = ag.lp.EllipticalSersic(
            centre=(0.1, 0.1),
            elliptical_comps=(0.096225, -0.055555),
            intensity=0.3,
            effective_radius=1.0,
            sersic_index=2.5,
        )

        light.image_from_grid(grid=grid)

    def test__grid_interpolate_in__interpolates_based_on_intepolate_config(self):
        # `False` in interpolate.ini

        mask = ag.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = ag.Grid2D.from_mask(mask=mask)

        grid_interpolate = ag.Grid2DInterpolate.from_mask(
            mask=mask, pixel_scales_interp=0.1
        )

        light_profile = ag.lp.EllipticalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=grid)
        image_no_interpolate = light_profile.image_from_grid(grid=grid_interpolate)

        assert (image == image_no_interpolate).all()

        # `False` in interpolate.ini

        light_profile = ag.lp.SphericalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=grid)
        image_interpolate = light_profile.image_from_grid(grid=grid_interpolate)
        assert (image != image_interpolate).all()

        array_interp = light_profile.image_from_grid(grid=grid_interpolate.grid_interp)
        interpolated_array = grid_interpolate.interpolated_array_from_array_interp(
            array_interp=array_interp
        )
        assert (image_interpolate == interpolated_array).all()


class TestRegression:
    def test__centre_of_profile_in_right_place(self):
        grid = ag.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

        light_profile = ag.lp.EllipticalSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_from_grid(grid=grid)
        max_indexes = np.unravel_index(image.native.argmax(), image.shape_native)
        assert max_indexes == (1, 4)

        light_profile = ag.lp.SphericalSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_from_grid(grid=grid)
        max_indexes = np.unravel_index(image.native.argmax(), image.shape_native)
        assert max_indexes == (1, 4)

        grid = ag.Grid2DIterate.uniform(
            shape_native=(7, 7),
            pixel_scales=1.0,
            fractional_accuracy=0.99,
            sub_steps=[2, 4],
        )

        light_profile = ag.lp.EllipticalSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_from_grid(grid=grid)
        max_indexes = np.unravel_index(image.native.argmax(), image.shape_native)
        assert max_indexes == (1, 4)

        light_profile = ag.lp.SphericalSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_from_grid(grid=grid)
        max_indexes = np.unravel_index(image.native.argmax(), image.shape_native)
        assert max_indexes == (1, 4)


class TestGrids:
    def test__grid_to_eccentric_radius(self):
        elliptical = ag.lp.EllipticalSersic(elliptical_comps=(0.0, 0.333333))

        radii_0 = elliptical.grid_to_eccentric_radii(np.array([[1, 1]]))

        radii_1 = elliptical.grid_to_eccentric_radii(np.array([[-1, -1]]))

        assert radii_0 == pytest.approx(radii_1, 1e-10)
