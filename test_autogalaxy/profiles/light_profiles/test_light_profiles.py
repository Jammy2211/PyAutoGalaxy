from __future__ import division, print_function
import math
import numpy as np
import pytest
import scipy.special

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestRegression:
    def test__centre_of_profile_in_right_place(self):
        grid = ag.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

        light_profile = ag.lp.EllSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_2d_from(grid=grid)
        max_indexes = np.unravel_index(image.native.argmax(), image.shape_native)
        assert max_indexes == (1, 4)

        light_profile = ag.lp.SphSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_2d_from(grid=grid)
        max_indexes = np.unravel_index(image.native.argmax(), image.shape_native)
        assert max_indexes == (1, 4)

        grid = ag.Grid2DIterate.uniform(
            shape_native=(7, 7),
            pixel_scales=1.0,
            fractional_accuracy=0.99,
            sub_steps=[2, 4],
        )

        light_profile = ag.lp.EllSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_2d_from(grid=grid)
        max_indexes = np.unravel_index(image.native.argmax(), image.shape_native)
        assert max_indexes == (1, 4)

        light_profile = ag.lp.SphSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_2d_from(grid=grid)
        max_indexes = np.unravel_index(image.native.argmax(), image.shape_native)
        assert max_indexes == (1, 4)


class TestImage1DFrom:
    def test__grid_2d_in__returns_1d_image_via_projected_quantities(self):

        grid_2d = ag.Grid2D.uniform(shape_native=(5, 5), pixel_scales=1.0)

        gaussian = ag.lp.EllGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
        )

        image_1d = gaussian.image_1d_from(grid=grid_2d)
        image_2d = gaussian.image_2d_from(grid=grid_2d)

        assert image_1d[0] == pytest.approx(image_2d.native[2, 2], 1.0e-4)
        assert image_1d[1] == pytest.approx(image_2d.native[2, 3], 1.0e-4)
        assert image_1d[2] == pytest.approx(image_2d.native[2, 4], 1.0e-4)

        gaussian = ag.lp.EllGaussian(
            centre=(0.2, 0.2), elliptical_comps=(0.3, 0.3), intensity=1.0, sigma=1.0
        )

        image_1d = gaussian.image_1d_from(grid=grid_2d)

        grid_2d_projected = grid_2d.grid_2d_radial_projected_from(
            centre=gaussian.centre, angle=gaussian.angle + 90.0
        )

        image_projected = gaussian.image_2d_from(grid=grid_2d_projected)

        assert image_1d == pytest.approx(image_projected, 1.0e-4)
        assert (image_1d.grid_radial == np.array([0.0, 1.0, 2.0])).all()


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

        sersic = ag.lp.SphSersic(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=0.5, profile=sersic
        )

        luminosity_integral = sersic.luminosity_within_circle_from(radius=0.5)

        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

        luminosity_grid = luminosity_from_radius_and_profile(radius=1.0, profile=sersic)

        luminosity_integral = sersic.luminosity_within_circle_from(radius=1.0)

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

        light_profile = ag.lp.EllSersic(intensity=1.0)

        image = light_profile.image_2d_from(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = ag.Grid2D.from_mask(mask=mask_sub_2)
        image_sub_2 = light_profile.image_2d_from(grid=grid_sub_2).binned

        assert (image == image_sub_2).all()

        grid = ag.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.95, sub_steps=[2, 4, 8]
        )

        light_profile = ag.lp.EllSersic(centre=(0.08, 0.08), intensity=1.0)

        image = light_profile.image_2d_from(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from(mask=mask, sub_size=4)
        grid_sub_4 = ag.Grid2D.from_mask(mask=mask_sub_4)
        image_sub_4 = light_profile.image_2d_from(grid=grid_sub_4).binned

        assert image[0] == image_sub_4[0]

        mask_sub_8 = mask.mask_new_sub_size_from(mask=mask, sub_size=8)
        grid_sub_8 = ag.Grid2D.from_mask(mask=mask_sub_8)
        image_sub_8 = light_profile.image_2d_from(grid=grid_sub_8).binned

        assert image[4] == image_sub_8[4]

    def test__grid_iterate_in__iterates_grid_correctly_for_peak(self):
        grid = ag.Grid2DIterate.uniform(
            shape_native=(100, 100),
            pixel_scales=0.1,
            fractional_accuracy=0.9999,
            sub_steps=[2, 4, 8, 16, 24],
        )

        light = ag.lp.EllSersic(
            centre=(0.1, 0.1),
            elliptical_comps=(0.096225, -0.055555),
            intensity=0.3,
            effective_radius=1.0,
            sersic_index=2.5,
        )

        light.image_2d_from(grid=grid)


class TestGaussian:
    def test__image_2d_from(self):
        gaussian = ag.lp.EllGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
        )

        image = gaussian.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(0.60653, 1e-2)

        gaussian = ag.lp.EllGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=2.0, sigma=1.0
        )

        image = gaussian.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(2.0 * 0.60653, 1e-2)

        gaussian = ag.lp.EllGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
        )

        image = gaussian.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(0.88249, 1e-2)

        gaussian = ag.lp.EllGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
        )

        image = gaussian.image_2d_from(grid=np.array([[0.0, 3.0]]))

        assert image == pytest.approx(0.3246, 1e-2)

        elliptical = ag.lp.EllGaussian(
            elliptical_comps=(0.0, 0.0), intensity=3.0, sigma=2.0
        )
        spherical = ag.lp.SphGaussian(intensity=3.0, sigma=2.0)

        image_elliptical = elliptical.image_2d_from(grid=grid)
        image_spherical = spherical.image_2d_from(grid=grid)

        assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)


class TestMoffat:
    def test__image_2d_from(self):
        moffat = ag.lp.EllMoffat(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            alpha=1.0,
            beta=1.0,
        )

        image = moffat.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(0.5, 1e-4)

        moffat = ag.lp.EllMoffat(
            centre=(0.0, 0.0),
            elliptical_comps=(0.2, 0.2),
            intensity=1.0,
            alpha=1.8,
            beta=0.75,
        )

        image = moffat.image_2d_from(grid=np.array([[0.0, 2.0]]))

        assert image == pytest.approx(0.7340746, 1e-4)

        moffat = ag.lp.SphMoffat(centre=(0.0, 0.0), intensity=1.0, alpha=1.8, beta=0.75)

        image = moffat.image_2d_from(grid=np.array([[0.0, 2.0]]))

        assert image == pytest.approx(0.5471480213, 1e-4)


class TestSersic:
    def test__image_2d_from(self):
        sersic = ag.lp.EllSersic(
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
        )

        image = sersic.image_2d_from(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(0.351797, 1e-3)

        sersic = ag.lp.EllSersic(
            elliptical_comps=(0.0, 0.0),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )
        # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797

        image = sersic.image_2d_from(grid=np.array([[1.5, 0.0]]))

        assert image == pytest.approx(4.90657319276, 1e-3)

        sersic = ag.lp.EllSersic(
            elliptical_comps=(0.0, 0.333333),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )

        image = sersic.image_2d_from(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(5.38066670129, 1e-3)

        elliptical = ag.lp.EllSersic(
            elliptical_comps=(0.0, 0.0),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
        )

        spherical = ag.lp.SphSersic(
            intensity=3.0, effective_radius=2.0, sersic_index=2.0
        )

        image_elliptical = elliptical.image_2d_from(grid=grid)

        image_spherical = spherical.image_2d_from(grid=grid)

        assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)


class TestExponential:
    def test__image_2d_from(self):
        exponential = ag.lp.EllExponential(
            elliptical_comps=(0.0, 0.333333), intensity=3.0, effective_radius=2.0
        )

        image = exponential.image_2d_from(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(4.9047, 1e-3)

        exponential = ag.lp.EllExponential(
            elliptical_comps=(0.0, -0.333333), intensity=2.0, effective_radius=3.0
        )

        image = exponential.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(4.8566, 1e-3)

        exponential = ag.lp.EllExponential(
            elliptical_comps=(0.0, -0.333333), intensity=4.0, effective_radius=3.0
        )

        image = exponential.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(2.0 * 4.8566, 1e-3)

        value = exponential.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert value == pytest.approx(2.0 * 4.8566, 1e-3)

        elliptical = ag.lp.EllExponential(
            elliptical_comps=(0.0, 0.0), intensity=3.0, effective_radius=2.0
        )

        spherical = ag.lp.SphExponential(intensity=3.0, effective_radius=2.0)

        image_elliptical = elliptical.image_2d_from(grid=grid)
        image_spherical = spherical.image_2d_from(grid=grid)

        assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)


class TestDevVaucouleurs:
    def test__image_2d_from(self):
        dev_vaucouleurs = ag.lp.EllDevVaucouleurs(
            elliptical_comps=(0.0, 0.333333), intensity=3.0, effective_radius=2.0
        )

        image = dev_vaucouleurs.image_2d_from(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(5.6697, 1e-3)

        dev_vaucouleurs = ag.lp.EllDevVaucouleurs(
            elliptical_comps=(0.0, -0.333333), intensity=2.0, effective_radius=3.0
        )

        image = dev_vaucouleurs.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(7.4455, 1e-3)

        dev_vaucouleurs = ag.lp.EllDevVaucouleurs(
            elliptical_comps=(0.0, -0.333333), intensity=4.0, effective_radius=3.0
        )

        image = dev_vaucouleurs.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(2.0 * 7.4455, 1e-3)

        value = dev_vaucouleurs.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert value == pytest.approx(2.0 * 7.4455, 1e-3)

        elliptical = ag.lp.EllDevVaucouleurs(
            elliptical_comps=(0.0, 0.0), intensity=3.0, effective_radius=2.0
        )

        spherical = ag.lp.SphDevVaucouleurs(intensity=3.0, effective_radius=2.0)

        image_elliptical = elliptical.image_2d_from(grid=grid)

        image_spherical = spherical.image_2d_from(grid=grid)

        assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)


class TestSersicCore:
    def test__image_2d_from(self):

        core_sersic = ag.lp.EllSersicCore(
            elliptical_comps=(0.0, 0.333333),
            effective_radius=5.0,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=1.0,
        )

        image = core_sersic.image_2d_from(grid=np.array([[0.0, 0.1]]))

        assert image == pytest.approx(0.0255173, 1.0e-4)

        elliptical = ag.lp.EllSersicCore(
            elliptical_comps=(0.0, 0.0),
            effective_radius=5.0,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=1.0,
        )

        spherical = ag.lp.SphSersicCore(
            effective_radius=5.0,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=1.0,
        )

        image_elliptical = elliptical.image_2d_from(grid=grid)

        image_spherical = spherical.image_2d_from(grid=grid)

        assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)


class TestChameleon:
    def test__image_2d_from(self):

        chameleon = ag.lp.EllChameleon(
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
        )

        image = chameleon.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(0.018605, 1e-3)

        chameleon = ag.lp.EllChameleon(
            elliptical_comps=(0.5, 0.0),
            intensity=3.0,
            core_radius_0=0.2,
            core_radius_1=0.4,
        )
        # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797

        image = chameleon.image_2d_from(grid=np.array([[0.0, 1.5]]))

        assert image == pytest.approx(0.0078149, 1e-3)

        chameleon = ag.lp.EllChameleon(
            elliptical_comps=(0.0, 0.333333),
            intensity=3.0,
            core_radius_0=0.2,
            core_radius_1=0.4,
        )

        image = chameleon.image_2d_from(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(0.024993, 1e-3)

        elliptical = ag.lp.EllChameleon(
            elliptical_comps=(0.0, 0.0),
            intensity=3.0,
            core_radius_0=0.2,
            core_radius_1=0.4,
        )

        spherical = ag.lp.SphChameleon(
            intensity=3.0, core_radius_0=0.2, core_radius_1=0.4
        )

        image_elliptical = elliptical.image_2d_from(grid=grid)

        image_spherical = spherical.image_2d_from(grid=grid)

        assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)


class TestEff:
    def test__image_2d_from(self):
        eff = ag.lp.EllEff(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
        )

        image = eff.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(0.35355, 1e-2)

        eff = ag.lp.EllEff(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=2.0,
            effective_radius=1.0,
        )

        image = eff.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(2.0 * 0.35355, 1e-2)

        eff = ag.lp.EllEff(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=2.0,
        )

        image = eff.image_2d_from(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(0.71554, 1e-2)

        eff = ag.lp.EllEff(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=2.0,
        )

        image = eff.image_2d_from(grid=np.array([[0.0, 3.0]]))

        assert image == pytest.approx(0.17067, 1e-2)

        elliptical = ag.lp.EllEff(
            elliptical_comps=(0.0, 0.0), intensity=3.0, effective_radius=2.0
        )
        spherical = ag.lp.SphEff(intensity=3.0, effective_radius=2.0)

        image_elliptical = elliptical.image_2d_from(grid=grid)
        image_spherical = spherical.image_2d_from(grid=grid)

        assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)

    def test__half_light_radius(self):

        eff = ag.lp.EllEff(effective_radius=2.0, eta=4.0)

        assert eff.half_light_radius == pytest.approx(1.01964, 1e-2)
