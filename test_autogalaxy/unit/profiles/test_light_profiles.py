from __future__ import division, print_function

import math
import os

from autoconf import conf
import autogalaxy as ag
import numpy as np
import pytest
import scipy.special
from test_autogalaxy import mock

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestGaussian:
    def test__constructor_and_units(self):

        gaussian = ag.lp.EllipticalGaussian(
            centre=(1.0, 2.0),
            elliptical_comps=(0.333333, 0.0),
            intensity=1.0,
            sigma=0.1,
        )

        assert gaussian.centre == (1.0, 2.0)
        assert isinstance(gaussian.centre[0], ag.dim.Length)
        assert isinstance(gaussian.centre[1], ag.dim.Length)
        assert gaussian.centre[0].unit == "arcsec"
        assert gaussian.centre[1].unit == "arcsec"

        assert gaussian.axis_ratio == pytest.approx(0.5, 1.0e-4)
        assert isinstance(gaussian.axis_ratio, float)

        assert gaussian.phi == pytest.approx(45.0, 1.0e-4)
        assert isinstance(gaussian.phi, float)

        assert gaussian.intensity == 1.0
        assert isinstance(gaussian.intensity, ag.dim.Luminosity)
        assert gaussian.intensity.unit == "eps"

        assert gaussian.sigma == 0.1
        assert isinstance(gaussian.sigma, ag.dim.Length)
        assert gaussian.sigma.unit_length == "arcsec"

        gaussian = ag.lp.SphericalGaussian(centre=(1.0, 2.0), intensity=1.0, sigma=0.1)

        assert gaussian.centre == (1.0, 2.0)
        assert isinstance(gaussian.centre[0], ag.dim.Length)
        assert isinstance(gaussian.centre[1], ag.dim.Length)
        assert gaussian.centre[0].unit == "arcsec"
        assert gaussian.centre[1].unit == "arcsec"

        assert gaussian.axis_ratio == 1.0
        assert isinstance(gaussian.axis_ratio, float)

        assert gaussian.phi == 0.0
        assert isinstance(gaussian.phi, float)

        assert gaussian.intensity == 1.0
        assert isinstance(gaussian.intensity, ag.dim.Luminosity)
        assert gaussian.intensity.unit == "eps"

        assert gaussian.sigma == 0.1
        assert isinstance(gaussian.sigma, ag.dim.Length)
        assert gaussian.sigma.unit_length == "arcsec"

    def test__intensity_as_radius__correct_value(self):

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
        )

        image = gaussian.image_from_grid_radii(grid_radii=1.0)

        assert image == pytest.approx(0.24197, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=2.0, sigma=1.0
        )

        image = gaussian.image_from_grid_radii(grid_radii=1.0)

        assert image == pytest.approx(2.0 * 0.24197, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
        )

        image = gaussian.image_from_grid_radii(grid_radii=1.0)

        assert image == pytest.approx(0.1760, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
        )

        image = gaussian.image_from_grid_radii(grid_radii=3.0)

        assert image == pytest.approx(0.0647, 1e-2)

    def test__intensity_from_grid__same_values_as_above(self):
        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
        )

        image = gaussian.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(0.24197, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=2.0, sigma=1.0
        )

        image = gaussian.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(2.0 * 0.24197, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
        )

        image = gaussian.image_from_grid(grid=np.array([[0.0, 1.0]]))

        assert image == pytest.approx(0.1760, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
        )

        image = gaussian.image_from_grid(grid=np.array([[0.0, 3.0]]))

        assert image == pytest.approx(0.0647, 1e-2)

        value = gaussian.image_from_grid(grid=np.array([[0.0, 3.0]]))

        assert value == pytest.approx(0.0647, 1e-2)

    def test__intensity_from_grid__change_geometry(self):

        gaussian = ag.lp.EllipticalGaussian(
            centre=(1.0, 1.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
        )
        image = gaussian.image_from_grid(grid=np.array([[1.0, 0.0]]))
        assert image == pytest.approx(0.24197, 1e-2)

        gaussian = ag.lp.EllipticalGaussian(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.333333),
            intensity=1.0,
            sigma=1.0,
        )

        image = gaussian.image_from_grid(grid=np.array([[1.0, 0.0]]))

        assert image == pytest.approx(0.05399, 1e-2)

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

        elliptical_image = elliptical.image_from_grid(grid=grid)
        spherical_image = spherical.image_from_grid(grid=grid)

        assert (elliptical_image == spherical_image).all()

    def test__output_image_is_array(self):
        grid = ag.Grid.uniform(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)

        gaussian = ag.lp.EllipticalGaussian()

        image = gaussian.image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)

        gaussian = ag.lp.SphericalGaussian()

        image = gaussian.image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)


class TestSersic:
    def test__constructor_and_units(self):

        sersic = ag.lp.EllipticalSersic(
            centre=(1.0, 2.0),
            elliptical_comps=(0.333333, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], ag.dim.Length)
        assert isinstance(sersic.centre[1], ag.dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == pytest.approx(0.5, 1.0e-4)
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == pytest.approx(45.0, 1.0e-4)
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, ag.dim.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, ag.dim.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == pytest.approx(
            0.6 / np.sqrt(0.5), 1.0e-4
        )

        sersic = ag.lp.SphericalSersic(
            centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6, sersic_index=4.0
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], ag.dim.Length)
        assert isinstance(sersic.centre[1], ag.dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, ag.dim.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, ag.dim.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__intensity_at_radius__correct_value(self):

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

    def test__intensity_from_grid__correct_values_for_input_parameters(self):

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

        assert (image_elliptical == image_spherical).all()

    def test__summarize_in_units(self):

        test_path = "{}/config/summary".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        conf.instance = conf.Config(config_path=test_path)

        sersic = ag.lp.SphericalSersic(
            intensity=3.0, effective_radius=2.0, sersic_index=2.0
        )

        summary_text = sersic.summarize_in_units(
            radii=[ag.dim.Length(10.0), ag.dim.Length(500.0)],
            prefix="sersic_",
            unit_length="arcsec",
            unit_luminosity="eps",
            whitespace=50,
        )

        i = 0

        assert summary_text[i] == "Light Profile = SphericalSersic\n"
        i += 1
        assert (
            summary_text[i]
            == "sersic_luminosity_within_10.00_arcsec             1.8854e+02 eps"
        )
        i += 1
        assert (
            summary_text[i]
            == "sersic_luminosity_within_500.00_arcsec            1.9573e+02 eps"
        )
        i += 1

    def test__output_image_is_autoarray(self):
        grid = ag.Grid.uniform(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)

        sersic = ag.lp.EllipticalSersic()

        image = sersic.image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)

        sersic = ag.lp.SphericalSersic()

        image = sersic.image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)


class TestExponential:
    def test__constructor_and_units(self):
        exponential = ag.lp.EllipticalExponential(
            centre=(1.0, 2.0),
            elliptical_comps=(0.333333, 0.0),
            intensity=1.0,
            effective_radius=0.6,
        )

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], ag.dim.Length)
        assert isinstance(exponential.centre[1], ag.dim.Length)
        assert exponential.centre[0].unit == "arcsec"
        assert exponential.centre[1].unit == "arcsec"

        assert exponential.axis_ratio == pytest.approx(0.5, 1.0e-4)
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == pytest.approx(45.0, 1.0e-4)
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, ag.dim.Luminosity)
        assert exponential.intensity.unit == "eps"

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, ag.dim.Length)
        assert exponential.effective_radius.unit_length == "arcsec"

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == pytest.approx(
            0.6 / np.sqrt(0.5), 1.0e-4
        )

        exponential = ag.lp.SphericalExponential(
            centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6
        )

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], ag.dim.Length)
        assert isinstance(exponential.centre[1], ag.dim.Length)
        assert exponential.centre[0].unit == "arcsec"
        assert exponential.centre[1].unit == "arcsec"

        assert exponential.axis_ratio == 1.0
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 0.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, ag.dim.Luminosity)
        assert exponential.intensity.unit == "eps"

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, ag.dim.Length)
        assert exponential.effective_radius.unit_length == "arcsec"

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6

    def test__intensity_at_radius__correct_value(self):
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

    def test__intensity_from_grid__correct_values(self):
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

    def test__intensity_from_grid__change_geometry(self):
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

        elliptical_image = elliptical.image_from_grid(grid=grid)
        spherical_image = spherical.image_from_grid(grid=grid)

        assert (elliptical_image == spherical_image).all()

    def test__output_image_is_autoarray(self):
        grid = ag.Grid.uniform(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)

        exponential = ag.lp.EllipticalExponential()

        image = exponential.image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)

        exponential = ag.lp.SphericalExponential()

        image = exponential.image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)


class TestDevVaucouleurs:
    def test__constructor_and_units(self):
        dev_vaucouleurs = ag.lp.EllipticalDevVaucouleurs(
            centre=(1.0, 2.0),
            elliptical_comps=(0.333333, 0.0),
            intensity=1.0,
            effective_radius=0.6,
        )

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], ag.dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], ag.dim.Length)
        assert dev_vaucouleurs.centre[0].unit == "arcsec"
        assert dev_vaucouleurs.centre[1].unit == "arcsec"

        assert dev_vaucouleurs.axis_ratio == pytest.approx(0.5, 1.0e-4)
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == pytest.approx(45.0, 1.0e-4)
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, ag.dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == "eps"

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, ag.dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == "arcsec"

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == pytest.approx(
            0.6 / np.sqrt(0.5), 1.0e-4
        )

        dev_vaucouleurs = ag.lp.SphericalDevVaucouleurs(
            centre=(1.0, 2.0), intensity=1.0, effective_radius=0.6
        )

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], ag.dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], ag.dim.Length)
        assert dev_vaucouleurs.centre[0].unit == "arcsec"
        assert dev_vaucouleurs.centre[1].unit == "arcsec"

        assert dev_vaucouleurs.axis_ratio == 1.0
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 0.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, ag.dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == "eps"

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, ag.dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == "arcsec"

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6

    def test__intensity_at_radius__correct_value(self):
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

    def test__intensity_from_grid__correct_values(self):
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

    def test__intensity_from_grid__change_geometry(self):
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

        elliptical_image = elliptical.image_from_grid(grid=grid)

        spherical_image = spherical.image_from_grid(grid=grid)

        assert (elliptical_image == spherical_image).all()

    def test__output_image_is_autoarray(self):
        grid = ag.Grid.uniform(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)

        dev_vaucouleurs = ag.lp.EllipticalDevVaucouleurs()

        image = dev_vaucouleurs.image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)

        dev_vaucouleurs = ag.lp.SphericalDevVaucouleurs()

        image = dev_vaucouleurs.image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)


class TestCoreSersic:
    def test__constructor_and_units(self):
        core_sersic = ag.lp.EllipticalCoreSersic(
            centre=(1.0, 2.0),
            elliptical_comps=(0.333333, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=2.0,
        )

        assert core_sersic.centre == (1.0, 2.0)
        assert isinstance(core_sersic.centre[0], ag.dim.Length)
        assert isinstance(core_sersic.centre[1], ag.dim.Length)
        assert core_sersic.centre[0].unit == "arcsec"
        assert core_sersic.centre[1].unit == "arcsec"

        assert core_sersic.axis_ratio == pytest.approx(0.5, 1.0e-4)
        assert isinstance(core_sersic.axis_ratio, float)

        assert core_sersic.phi == pytest.approx(45.0, 1.0e-4)
        assert isinstance(core_sersic.phi, float)

        assert core_sersic.intensity == 1.0
        assert isinstance(core_sersic.intensity, ag.dim.Luminosity)
        assert core_sersic.intensity.unit == "eps"

        assert core_sersic.effective_radius == 0.6
        assert isinstance(core_sersic.effective_radius, ag.dim.Length)
        assert core_sersic.effective_radius.unit_length == "arcsec"

        assert core_sersic.sersic_index == 4.0
        assert isinstance(core_sersic.sersic_index, float)

        assert core_sersic.radius_break == 0.01
        assert isinstance(core_sersic.radius_break, ag.dim.Length)
        assert core_sersic.radius_break.unit_length == "arcsec"

        assert core_sersic.intensity_break == 0.1
        assert isinstance(core_sersic.intensity_break, ag.dim.Luminosity)
        assert core_sersic.intensity_break.unit == "eps"

        assert core_sersic.gamma == 1.0
        assert isinstance(core_sersic.gamma, float)

        assert core_sersic.alpha == 2.0
        assert isinstance(core_sersic.alpha, float)

        assert core_sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic.elliptical_effective_radius == pytest.approx(
            0.6 / np.sqrt(0.5), 1.0e-4
        )

        core_sersic = ag.lp.SphericalCoreSersic(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            radius_break=0.01,
            intensity_break=0.1,
            gamma=1.0,
            alpha=2.0,
        )

        assert core_sersic.centre == (1.0, 2.0)
        assert isinstance(core_sersic.centre[0], ag.dim.Length)
        assert isinstance(core_sersic.centre[1], ag.dim.Length)
        assert core_sersic.centre[0].unit == "arcsec"
        assert core_sersic.centre[1].unit == "arcsec"

        assert core_sersic.axis_ratio == 1.0
        assert isinstance(core_sersic.axis_ratio, float)

        assert core_sersic.phi == 0.0
        assert isinstance(core_sersic.phi, float)

        assert core_sersic.intensity == 1.0
        assert isinstance(core_sersic.intensity, ag.dim.Luminosity)
        assert core_sersic.intensity.unit == "eps"

        assert core_sersic.effective_radius == 0.6
        assert isinstance(core_sersic.effective_radius, ag.dim.Length)
        assert core_sersic.effective_radius.unit_length == "arcsec"

        assert core_sersic.sersic_index == 4.0
        assert isinstance(core_sersic.sersic_index, float)

        assert core_sersic.radius_break == 0.01
        assert isinstance(core_sersic.radius_break, ag.dim.Length)
        assert core_sersic.radius_break.unit_length == "arcsec"

        assert core_sersic.intensity_break == 0.1
        assert isinstance(core_sersic.intensity_break, ag.dim.Luminosity)
        assert core_sersic.intensity_break.unit == "eps"

        assert core_sersic.gamma == 1.0
        assert isinstance(core_sersic.gamma, float)

        assert core_sersic.alpha == 2.0
        assert isinstance(core_sersic.alpha, float)

        assert core_sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert core_sersic.elliptical_effective_radius == 0.6

    def test__intensity_at_radius__correct_value(self):
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

        elliptical_image = elliptical.image_from_grid(grid=grid)

        spherical_image = spherical.image_from_grid(grid=grid)

        assert (elliptical_image == spherical_image).all()

    def test__output_image_is_autoarray(self):
        grid = ag.Grid.uniform(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)

        core_sersic = ag.lp.EllipticalCoreSersic()

        image = core_sersic.image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)

        core_sersic = ag.lp.SphericalCoreSersic()

        image = core_sersic.image_from_grid(grid=grid)

        assert image.shape_2d == (2, 2)


class TestBlurredProfileImages:
    def test__blurred_image_from_grid_and_psf(
        self, sub_grid_7x7, blurring_grid_7x7, psf_3x3, convolver_7x7
    ):

        light_profile = ag.lp.EllipticalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=sub_grid_7x7)

        blurring_image = light_profile.image_from_grid(grid=blurring_grid_7x7)

        blurred_image = convolver_7x7.convolved_image_from_image_and_blurring_image(
            image=image.in_1d_binned, blurring_image=blurring_image.in_1d_binned
        )

        light_profile_blurred_image = light_profile.blurred_image_from_grid_and_psf(
            grid=sub_grid_7x7, blurring_grid=blurring_grid_7x7, psf=psf_3x3
        )

        assert blurred_image.in_1d == pytest.approx(
            light_profile_blurred_image.in_1d, 1.0e-4
        )
        assert blurred_image.in_2d == pytest.approx(
            light_profile_blurred_image.in_2d, 1.0e-4
        )

    def test__blurred_image_from_grid_and_convolver(
        self, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
    ):

        light_profile = ag.lp.EllipticalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=sub_grid_7x7)

        blurring_image = light_profile.image_from_grid(grid=blurring_grid_7x7)

        blurred_image = convolver_7x7.convolved_image_from_image_and_blurring_image(
            image=image.in_1d_binned, blurring_image=blurring_image.in_1d_binned
        )

        light_profile_blurred_image = light_profile.blurred_image_from_grid_and_convolver(
            grid=sub_grid_7x7, convolver=convolver_7x7, blurring_grid=blurring_grid_7x7
        )

        assert blurred_image.in_1d == pytest.approx(
            light_profile_blurred_image.in_1d, 1.0e-4
        )
        assert blurred_image.in_2d == pytest.approx(
            light_profile_blurred_image.in_2d, 1.0e-4
        )


class TestVisibilities:
    def test__visibilities_from_grid_and_transformer(
        self, grid_7x7, sub_grid_7x7, transformer_7x7_7
    ):
        light_profile = ag.lp.EllipticalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=grid_7x7)

        visibilities = transformer_7x7_7.visibilities_from_image(
            image=image.in_1d_binned
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
    def test__luminosity_in_eps__spherical_sersic_index_2__compare_to_analytic(self):
        sersic = ag.lp.SphericalSersic(
            intensity=3.0, effective_radius=2.0, sersic_index=2.0
        )

        radius = ag.dim.Length(0.5, "arcsec")

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic
        )

        luminosity_integral = sersic.luminosity_within_circle_in_units(
            radius=0.5, unit_luminosity="eps"
        )

        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

    def test__luminosity_in_eps__spherical_sersic_2__compare_to_grid(self):
        sersic = ag.lp.SphericalSersic(
            intensity=3.0, effective_radius=2.0, sersic_index=2.0
        )

        radius = ag.dim.Length(1.0, "arcsec")

        luminosity_grid = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic
        )

        luminosity_integral = sersic.luminosity_within_circle_in_units(
            radius=radius, unit_luminosity="eps"
        )

        assert luminosity_grid == pytest.approx(luminosity_integral, 0.02)

    def test__luminosity_units_conversions__uses_exposure_time(self):
        sersic_eps = ag.lp.SphericalSersic(
            intensity=ag.dim.Luminosity(3.0, "eps"),
            effective_radius=2.0,
            sersic_index=1.0,
        )

        radius = ag.dim.Length(0.5, "arcsec")

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic_eps
        )

        luminosity_integral = sersic_eps.luminosity_within_circle_in_units(
            radius=radius, unit_luminosity="eps", exposure_time=3.0
        )

        # eps -> eps

        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

        # eps -> counts

        luminosity_integral = sersic_eps.luminosity_within_circle_in_units(
            radius=radius, unit_luminosity="counts", exposure_time=3.0
        )

        assert 3.0 * luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

        sersic_counts = ag.lp.SphericalSersic(
            intensity=ag.dim.Luminosity(3.0, "counts"),
            effective_radius=2.0,
            sersic_index=1.0,
        )

        radius = ag.dim.Length(0.5, "arcsec")

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic_counts
        )
        luminosity_integral = sersic_counts.luminosity_within_circle_in_units(
            radius=radius, unit_luminosity="eps", exposure_time=3.0
        )

        # counts -> eps

        assert luminosity_analytic / 3.0 == pytest.approx(luminosity_integral, 1e-3)

        luminosity_integral = sersic_counts.luminosity_within_circle_in_units(
            radius=radius, unit_luminosity="counts", exposure_time=3.0
        )

        # counts -> counts

        assert luminosity_analytic == pytest.approx(luminosity_integral, 1e-3)

    def test__radius_units_conversions__light_profile_updates_units_and_computes_correct_luminosity(
        self
    ):
        cosmology = mock.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0)

        sersic_arcsec = ag.lp.SphericalSersic(
            centre=(ag.dim.Length(0.0, "arcsec"), ag.dim.Length(0.0, "arcsec")),
            intensity=ag.dim.Luminosity(3.0, "eps"),
            effective_radius=ag.dim.Length(2.0, "arcsec"),
            sersic_index=1.0,
        )

        sersic_kpc = ag.lp.SphericalSersic(
            centre=(ag.dim.Length(0.0, "kpc"), ag.dim.Length(0.0, "kpc")),
            intensity=ag.dim.Luminosity(3.0, "eps"),
            effective_radius=ag.dim.Length(4.0, "kpc"),
            sersic_index=1.0,
        )

        radius = ag.dim.Length(0.5, "arcsec")

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic_arcsec
        )

        # arcsec -> arcsec

        luminosity = sersic_arcsec.luminosity_within_circle_in_units(radius=radius)

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        # kpc -> arcsec

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=1.0, profile=sersic_kpc
        )

        luminosity = sersic_kpc.luminosity_within_circle_in_units(
            radius=radius, redshift_object=0.5, cosmology=cosmology
        )

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        radius = ag.dim.Length(0.5, "kpc")

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=radius, profile=sersic_kpc
        )

        # kpc -> kpc

        luminosity = sersic_kpc.luminosity_within_circle_in_units(radius=radius)

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        # kpc -> arcsec

        luminosity_analytic = luminosity_from_radius_and_profile(
            radius=0.25, profile=sersic_arcsec
        )

        luminosity = sersic_arcsec.luminosity_within_circle_in_units(
            radius=radius, redshift_object=0.5, cosmology=cosmology
        )

        assert luminosity_analytic == pytest.approx(luminosity, 1e-3)

        radius = ag.dim.Length(2.0, "arcsec")
        luminosity_arcsec = sersic_arcsec.luminosity_within_circle_in_units(
            radius=radius, redshift_object=0.5, unit_mass="angular", cosmology=cosmology
        )
        radius = ag.dim.Length(4.0, "kpc")
        luminosity_kpc = sersic_arcsec.luminosity_within_circle_in_units(
            radius=radius, redshift_object=0.5, unit_mass="angular", cosmology=cosmology
        )
        assert luminosity_arcsec == luminosity_kpc


class TestDecorators:
    def test__grid_iterate_in__iterates_grid_correctly(self):

        mask = ag.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = ag.GridIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        light_profile = ag.lp.EllipticalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = ag.Grid.from_mask(mask=mask_sub_2)
        image_sub_2 = light_profile.image_from_grid(grid=grid_sub_2).in_1d_binned

        assert (image == image_sub_2).all()

        grid = ag.GridIterate.from_mask(
            mask=mask, fractional_accuracy=0.95, sub_steps=[2, 4, 8]
        )

        light_profile = ag.lp.EllipticalSersic(centre=(0.08, 0.08), intensity=1.0)

        image = light_profile.image_from_grid(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = ag.Grid.from_mask(mask=mask_sub_4)
        image_sub_4 = light_profile.image_from_grid(grid=grid_sub_4).in_1d_binned

        assert image[0] == image_sub_4[0]

        mask_sub_8 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)
        grid_sub_8 = ag.Grid.from_mask(mask=mask_sub_8)
        image_sub_8 = light_profile.image_from_grid(grid=grid_sub_8).in_1d_binned

        assert image[4] == image_sub_8[4]

    def test__grid_iterate_in__iterates_grid_correctly_for_peak(self):

        grid = ag.GridIterate.uniform(
            shape_2d=(100, 100),
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

        # False in interpolate.ini

        mask = ag.Mask.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = ag.Grid.from_mask(mask=mask)

        grid_interpolate = ag.GridInterpolate.from_mask(
            mask=mask, pixel_scales_interp=0.1
        )

        light_profile = ag.lp.EllipticalSersic(intensity=1.0)

        image = light_profile.image_from_grid(grid=grid)
        image_no_interpolate = light_profile.image_from_grid(grid=grid_interpolate)

        assert (image == image_no_interpolate).all()

        # False in interpolate.ini

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

        grid = ag.Grid.uniform(shape_2d=(7, 7), pixel_scales=1.0)

        light_profile = ag.lp.EllipticalSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_from_grid(grid=grid)
        max_indexes = np.unravel_index(image.in_2d.argmax(), image.shape_2d)
        assert max_indexes == (1, 4)

        light_profile = ag.lp.SphericalSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_from_grid(grid=grid)
        max_indexes = np.unravel_index(image.in_2d.argmax(), image.shape_2d)
        assert max_indexes == (1, 4)

        grid = ag.GridIterate.uniform(
            shape_2d=(7, 7),
            pixel_scales=1.0,
            fractional_accuracy=0.99,
            sub_steps=[2, 4],
        )

        light_profile = ag.lp.EllipticalSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_from_grid(grid=grid)
        max_indexes = np.unravel_index(image.in_2d.argmax(), image.shape_2d)
        assert max_indexes == (1, 4)

        light_profile = ag.lp.SphericalSersic(centre=(2.0, 1.0), intensity=1.0)
        image = light_profile.image_from_grid(grid=grid)
        max_indexes = np.unravel_index(image.in_2d.argmax(), image.shape_2d)
        assert max_indexes == (1, 4)


class TestGrids:
    def test__grid_to_eccentric_radius(self):
        elliptical = ag.lp.EllipticalSersic(elliptical_comps=(0.0, 0.333333))

        radii_0 = elliptical.grid_to_eccentric_radii(np.array([[1, 1]]))

        radii_1 = elliptical.grid_to_eccentric_radii(np.array([[-1, -1]]))

        assert radii_0 == pytest.approx(radii_1, 1e-10)
