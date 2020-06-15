from __future__ import division, print_function

import os

from autoconf import conf
import autogalaxy as ag

import numpy as np
import pytest

directory = os.path.dirname(os.path.realpath(__file__))

grid_10 = ag.Grid.manual_2d(grid=np.array([[[1.0, 0.0]]]), pixel_scales=1.0, sub_size=1)
grid_zero = ag.Grid.manual_2d(
    grid=np.array([[[0.0000000001, 0.0]]]), pixel_scales=1.0, sub_size=1
)


class TestGaussian:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        gaussian = ag.lp.EllipticalGaussian(centre=(0.0, 0.0))

        image_1 = gaussian.image_from_grid(grid=grid_10)
        image_0 = gaussian.image_from_grid(grid=grid_zero)

        assert image_0 == pytest.approx(image_1, 1.0e-4)

        gaussian = ag.lp.SphericalGaussian(centre=(0.0, 0.0))

        image_1 = gaussian.image_from_grid(grid=grid_10)
        image_0 = gaussian.image_from_grid(grid=grid_zero)
        assert image_0 == pytest.approx(image_1, 1.0e-4)


class TestSersic:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        sersic = ag.lp.EllipticalSersic(centre=(0.0, 0.0))

        image_1 = sersic.image_from_grid(grid=grid_10)
        image_0 = sersic.image_from_grid(grid=grid_zero)
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        sersic = ag.lp.SphericalSersic(centre=(0.0, 0.0))

        image_1 = sersic.image_from_grid(grid=grid_10)
        image_0 = sersic.image_from_grid(grid=grid_zero)
        assert image_0 == pytest.approx(image_1, 1.0e-4)


class TestExponential:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        exponential = ag.lp.EllipticalExponential(centre=(0.0, 0.0))

        image_1 = exponential.image_from_grid(grid=grid_10)
        image_0 = exponential.image_from_grid(grid=grid_zero)
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        exponential = ag.lp.SphericalExponential(centre=(0.0, 0.0))

        image_1 = exponential.image_from_grid(grid=grid_10)
        image_0 = exponential.image_from_grid(grid=grid_zero)
        assert image_0 == pytest.approx(image_1, 1.0e-4)


class TestDevVaucouleurs:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        dev_vaucouleurs = ag.lp.EllipticalDevVaucouleurs(centre=(0.0, 0.0))

        image_1 = dev_vaucouleurs.image_from_grid(grid=grid_10)
        image_0 = dev_vaucouleurs.image_from_grid(grid=grid_zero)
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        dev_vaucouleurs = ag.lp.SphericalDevVaucouleurs(centre=(0.0, 0.0))

        image_1 = dev_vaucouleurs.image_from_grid(grid=grid_10)
        image_0 = dev_vaucouleurs.image_from_grid(grid=grid_zero)
        assert image_0 == pytest.approx(image_1, 1.0e-4)


class TestCoreSersic:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        core_sersic = ag.lp.EllipticalCoreSersic(centre=(0.0, 0.0))

        image_1 = core_sersic.image_from_grid(grid=grid_10)
        image_0 = core_sersic.image_from_grid(grid=grid_zero)
        assert image_0 == pytest.approx(image_1, 1.0e-4)

        core_sersic = ag.lp.SphericalCoreSersic(centre=(0.0, 0.0))

        image_1 = core_sersic.image_from_grid(grid=grid_10)
        image_0 = core_sersic.image_from_grid(grid=grid_zero)
        assert image_0 == pytest.approx(image_1, 1.0e-4)


class TestPointMass:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        point_mass = ag.mp.PointMass(centre=(0.0, 0.0), einstein_radius=1.0)

        deflections_1 = point_mass.deflections_from_grid(grid=grid_10)
        deflections_0 = point_mass.deflections_from_grid(
            grid=np.array([[0.00000001, 0.0]])
        )
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestCoredPowerLaw:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        cored_power_law = ag.mp.EllipticalCoredPowerLaw(
            centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0
        )

        convergence_1 = cored_power_law.convergence_from_grid(grid=grid_10)
        convergence_0 = cored_power_law.convergence_from_grid(
            grid=np.array([[1e-8, 0.0]])
        )
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)


class TestPowerLaw:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        power_law = ag.mp.EllipticalPowerLaw(
            centre=(0.0, 0.0), einstein_radius=1.0, slope=2.0
        )

        convergence_1 = power_law.convergence_from_grid(grid=grid_10)
        convergence_0 = power_law.convergence_from_grid(grid=np.array([[1.0e-9, 0.0]]))
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)


class TestCoredIsothermal:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        cored_isothermal = ag.mp.EllipticalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        convergence_1 = cored_isothermal.convergence_from_grid(grid=grid_10)
        convergence_0 = cored_isothermal.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        cored_isothermal = ag.mp.SphericalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        convergence_1 = cored_isothermal.convergence_from_grid(grid=grid_10)
        convergence_0 = cored_isothermal.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        cored_isothermal = ag.mp.EllipticalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        potential_1 = cored_isothermal.potential_from_grid(grid=grid_10)
        potential_0 = cored_isothermal.potential_from_grid(grid=grid_zero)
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        cored_isothermal = ag.mp.SphericalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        potential_1 = cored_isothermal.potential_from_grid(grid=grid_10)
        potential_0 = cored_isothermal.potential_from_grid(grid=grid_zero)
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        cored_isothermal = ag.mp.EllipticalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        deflections_1 = cored_isothermal.deflections_from_grid(grid=grid_10)
        deflections_0 = cored_isothermal.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        cored_isothermal = ag.mp.SphericalCoredIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )

        deflections_1 = cored_isothermal.deflections_from_grid(grid=grid_10)
        deflections_0 = cored_isothermal.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestIsothermal:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        isothermal = ag.mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        convergence_1 = isothermal.convergence_from_grid(grid=grid_10)
        convergence_0 = isothermal.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        isothermal = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        convergence_1 = isothermal.convergence_from_grid(grid=grid_10)
        convergence_0 = isothermal.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        isothermal = ag.mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        potential_1 = isothermal.potential_from_grid(grid=grid_10)
        potential_0 = isothermal.potential_from_grid(grid=grid_zero)
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        isothermal = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        potential_1 = isothermal.potential_from_grid(grid=grid_10)
        potential_0 = isothermal.potential_from_grid(grid=grid_zero)
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        isothermal = ag.mp.EllipticalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        deflections_1 = isothermal.deflections_from_grid(grid=grid_10)
        deflections_0 = isothermal.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        isothermal = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

        deflections_1 = isothermal.deflections_from_grid(grid=grid_10)
        deflections_0 = isothermal.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestGeneralizedNFW:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        gnfw = ag.mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))

        convergence_1 = gnfw.convergence_from_grid(grid=grid_10)
        convergence_0 = gnfw.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        gnfw = ag.mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))

        potential_1 = gnfw.potential_from_grid(grid=grid_10)
        potential_0 = gnfw.potential_from_grid(grid=grid_zero)
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        gnfw = ag.mp.SphericalGeneralizedNFW(centre=(0.0, 0.0))

        deflections_1 = gnfw.deflections_from_grid(grid=grid_10)
        deflections_0 = gnfw.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestTruncatedNFW:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        truncated_nfw = ag.mp.SphericalTruncatedNFW(centre=(0.0, 0.0))

        convergence_1 = truncated_nfw.convergence_from_grid(grid=grid_10)
        convergence_0 = truncated_nfw.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        truncated_nfw = ag.mp.SphericalTruncatedNFW(centre=(0.0, 0.0))

        potential_1 = truncated_nfw.potential_from_grid(grid=grid_10)
        potential_0 = truncated_nfw.potential_from_grid(grid=grid_zero)
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        truncated_nfw = ag.mp.SphericalTruncatedNFW(centre=(0.0, 0.0))

        deflections_1 = truncated_nfw.deflections_from_grid(grid=grid_10)
        deflections_0 = truncated_nfw.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestNFW:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        nfw = ag.mp.EllipticalNFW(centre=(0.0, 0.0))

        convergence_1 = nfw.convergence_from_grid(grid=grid_10)
        convergence_0 = nfw.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        nfw = ag.mp.SphericalNFW(centre=(0.0, 0.0))

        convergence_1 = nfw.convergence_from_grid(grid=grid_10)
        convergence_0 = nfw.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        nfw = ag.mp.EllipticalNFW(centre=(0.0, 0.0))

        potential_1 = nfw.potential_from_grid(grid=grid_10)
        potential_0 = nfw.potential_from_grid(grid=grid_zero)
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        nfw = ag.mp.SphericalNFW(centre=(0.0, 0.0))

        potential_1 = nfw.potential_from_grid(grid=grid_10)
        potential_0 = nfw.potential_from_grid(grid=grid_zero)
        assert potential_0 == pytest.approx(potential_1, 1.0e-4)

        nfw = ag.mp.EllipticalNFW(centre=(0.0, 0.0))

        deflections_1 = nfw.deflections_from_grid(grid=grid_10)
        deflections_0 = nfw.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        nfw = ag.mp.SphericalNFW(centre=(0.0, 0.0))

        deflections_1 = nfw.deflections_from_grid(grid=grid_10)
        deflections_0 = nfw.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestSersicMass:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        sersic = ag.mp.EllipticalSersic(centre=(0.0, 0.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_10)
        convergence_0 = sersic.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = ag.mp.SphericalSersic(centre=(0.0, 0.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_10)
        convergence_0 = sersic.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = ag.mp.EllipticalSersic(centre=(0.0, 0.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_10)
        deflections_0 = sersic.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        sersic = ag.mp.SphericalSersic(centre=(0.0, 0.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_10)
        deflections_0 = sersic.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestExponentialMass:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        exponential = ag.mp.EllipticalExponential(centre=(0.0, 0.0))

        convergence_1 = exponential.convergence_from_grid(grid=grid_10)
        convergence_0 = exponential.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        exponential = ag.mp.SphericalExponential(centre=(0.0, 0.0))

        convergence_1 = exponential.convergence_from_grid(grid=grid_10)
        convergence_0 = exponential.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        exponential = ag.mp.EllipticalExponential(centre=(0.0, 0.0))

        deflections_1 = exponential.deflections_from_grid(grid=grid_10)
        deflections_0 = exponential.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        exponential = ag.mp.SphericalExponential(centre=(0.0, 0.0))

        deflections_1 = exponential.deflections_from_grid(grid=grid_10)
        deflections_0 = exponential.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestDevVaucouleursMass:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        dev_vaucouleurs = ag.mp.EllipticalDevVaucouleurs(centre=(0.0, 0.0))

        convergence_1 = dev_vaucouleurs.convergence_from_grid(grid=grid_10)
        convergence_0 = dev_vaucouleurs.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        dev_vaucouleurs = ag.mp.EllipticalDevVaucouleurs(centre=(0.0, 0.0))

        deflections_1 = dev_vaucouleurs.deflections_from_grid(grid=grid_10)
        deflections_0 = dev_vaucouleurs.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        dev_vaucouleurs = ag.mp.SphericalDevVaucouleurs(centre=(0.0, 0.0))

        deflections_1 = dev_vaucouleurs.deflections_from_grid(grid=grid_10)
        deflections_0 = dev_vaucouleurs.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestSersicMassRadialGradient:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        sersic = ag.mp.EllipticalSersicRadialGradient(centre=(0.0, 0.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_10)
        convergence_0 = sersic.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = ag.mp.SphericalSersicRadialGradient(centre=(0.0, 0.0))

        convergence_1 = sersic.convergence_from_grid(grid=grid_10)
        convergence_0 = sersic.convergence_from_grid(grid=grid_zero)
        assert convergence_0 == pytest.approx(convergence_1, 1.0e-4)

        sersic = ag.mp.EllipticalSersicRadialGradient(centre=(0.0, 0.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_10)
        deflections_0 = sersic.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)

        sersic = ag.mp.SphericalSersicRadialGradient(centre=(0.0, 0.0))

        deflections_1 = sersic.deflections_from_grid(grid=grid_10)
        deflections_0 = sersic.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestMassSheet:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0))

        deflections_1 = mass_sheet.deflections_from_grid(grid=grid_10)
        deflections_0 = mass_sheet.deflections_from_grid(grid=grid_zero)
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)


class TestExternalShear:
    def test__transform_grid_wrapper_and_move_radial_minimum_wrappers(self):

        conf.instance = conf.Config(config_path="{}/files/config/".format(directory))

        shear = ag.mp.ExternalShear(elliptical_comps=(0.1, 0.1))

        deflections_1 = shear.deflections_from_grid(grid=np.array([[1e-8, 0.0]]))
        deflections_0 = shear.deflections_from_grid(grid=np.array([[1e-9, 0.0]]))
        assert deflections_0 == pytest.approx(deflections_1, 1.0e-4)
