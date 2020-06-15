import autogalaxy as ag
import numpy as np
import pytest

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestGaussian(object):
    def test__constructor_and_units(self):
        gaussian = ag.lmp.EllipticalGaussian(
            centre=(1.0, 2.0),
            elliptical_comps=(0.333333, 0.0),
            intensity=1.0,
            sigma=4.0,
            mass_to_light_ratio=10.0,
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

        assert gaussian.sigma == 4.0
        assert isinstance(gaussian.sigma, ag.dim.Length)
        assert gaussian.sigma.unit_length == "arcsec"

        assert gaussian.mass_to_light_ratio == 10.0
        assert isinstance(gaussian.mass_to_light_ratio, ag.dim.MassOverLuminosity)
        assert gaussian.mass_to_light_ratio.unit == "angular / eps"

    def test__grid_calculations__same_as_gaussian(self):

        gaussian_lp = ag.lmp.EllipticalGaussian(
            elliptical_comps=(0.1, 0.05), intensity=1.0, sigma=5.0
        )
        gaussian_mp = ag.lmp.EllipticalGaussian(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            sigma=5.0,
            mass_to_light_ratio=2.0,
        )
        gaussian_lmp = ag.lmp.EllipticalGaussian(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            sigma=5.0,
            mass_to_light_ratio=2.0,
        )

        assert (
            gaussian_lp.image_from_grid(grid=grid)
            == gaussian_lmp.image_from_grid(grid=grid)
        ).all()
        assert (
            gaussian_mp.convergence_from_grid(grid=grid)
            == gaussian_lmp.convergence_from_grid(grid=grid)
        ).all()
        #    assert (sersic_mp.potential_from_grid(grid=grid) == sersic_lmp.potential_from_grid(grid=grid)).all()
        assert (
            gaussian_mp.deflections_from_grid(grid=grid)
            == gaussian_lmp.deflections_from_grid(grid=grid)
        ).all()


class TestSersic:
    def test__constructor_and_units(self):
        sersic = ag.lmp.EllipticalSersic(
            centre=(1.0, 2.0),
            elliptical_comps=(0.333333, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], ag.dim.Length)
        assert isinstance(sersic.centre[1], ag.dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.elliptical_comps == (0.333333, 0.0)

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

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, ag.dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == pytest.approx(
            0.6 / np.sqrt(0.5), 1.0e-4
        )

        sersic = ag.lmp.SphericalSersic(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
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

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, ag.dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__grid_calculations__same_as_sersic(self):
        sersic_lp = ag.lmp.EllipticalSersic(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
        )
        sersic_mp = ag.lmp.EllipticalSersic(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        sersic_lmp = ag.lmp.EllipticalSersic(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )

        assert (
            sersic_lp.image_from_grid(grid=grid)
            == sersic_lmp.image_from_grid(grid=grid)
        ).all()
        assert (
            sersic_mp.convergence_from_grid(grid=grid)
            == sersic_lmp.convergence_from_grid(grid=grid)
        ).all()
        #    assert (sersic_mp.potential_from_grid(grid=grid) == sersic_lmp.potential_from_grid(grid=grid)).all()
        assert (
            sersic_mp.deflections_from_grid(grid=grid)
            == sersic_lmp.deflections_from_grid(grid=grid)
        ).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllipticalSersic(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        spherical = ag.lmp.SphericalSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )

        assert (
            elliptical.image_from_grid(grid=grid)
            == spherical.image_from_grid(grid=grid)
        ).all()
        assert (
            elliptical.convergence_from_grid(grid=grid)
            == spherical.convergence_from_grid(grid=grid)
        ).all()
        # assert (elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)).all()
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )


class TestExponential:
    def test__constructor_and_units(self):
        exponential = ag.lmp.EllipticalExponential(
            centre=(1.0, 2.0),
            elliptical_comps=(0.333333, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
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

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, ag.dim.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == "angular / eps"

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == pytest.approx(
            0.6 / np.sqrt(0.5), 1.0e-4
        )

        exponential = ag.lmp.SphericalExponential(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
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

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, ag.dim.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == "angular / eps"

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6

    def test__grid_calculations__same_as_exponential(self):
        sersic_lp = ag.lmp.EllipticalExponential(
            elliptical_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6
        )
        sersic_mp = ag.lmp.EllipticalExponential(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
        )
        sersic_lmp = ag.lmp.EllipticalExponential(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
        )

        assert (
            sersic_lp.image_from_grid(grid=grid)
            == sersic_lmp.image_from_grid(grid=grid)
        ).all()
        assert (
            sersic_mp.convergence_from_grid(grid=grid)
            == sersic_lmp.convergence_from_grid(grid=grid)
        ).all()
        #    assert (sersic_mp.potential_from_grid(grid=grid) == sersic_lmp.potential_from_grid(grid=grid)).all()
        assert (
            sersic_mp.deflections_from_grid(grid=grid)
            == sersic_lmp.deflections_from_grid(grid=grid)
        ).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllipticalExponential(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
        )
        spherical = ag.lmp.SphericalExponential(
            centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
        )

        assert (
            elliptical.image_from_grid(grid=grid)
            == spherical.image_from_grid(grid=grid)
        ).all()
        assert (
            elliptical.convergence_from_grid(grid=grid)
            == spherical.convergence_from_grid(grid=grid)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )


class TestDevVaucouleurs:
    def test__constructor_and_units(self):
        dev_vaucouleurs = ag.lmp.EllipticalDevVaucouleurs(
            centre=(1.0, 2.0),
            elliptical_comps=(0.333333, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
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

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(
            dev_vaucouleurs.mass_to_light_ratio, ag.dim.MassOverLuminosity
        )
        assert dev_vaucouleurs.mass_to_light_ratio.unit == "angular / eps"

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == pytest.approx(
            0.6 / np.sqrt(0.5), 1.0e-4
        )

        dev_vaucouleurs = ag.lmp.SphericalDevVaucouleurs(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
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

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(
            dev_vaucouleurs.mass_to_light_ratio, ag.dim.MassOverLuminosity
        )
        assert dev_vaucouleurs.mass_to_light_ratio.unit == "angular / eps"

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6

    def test__grid_calculations__same_as_dev_vaucouleurs(self):
        sersic_lp = ag.lmp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6
        )
        sersic_mp = ag.lmp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
        )
        sersic_lmp = ag.lmp.EllipticalDevVaucouleurs(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
        )

        assert (
            sersic_lp.image_from_grid(grid=grid)
            == sersic_lmp.image_from_grid(grid=grid)
        ).all()
        assert (
            sersic_mp.convergence_from_grid(grid=grid)
            == sersic_lmp.convergence_from_grid(grid=grid)
        ).all()
        #    assert (sersic_mp.potential_from_grid(grid=grid) == sersic_lmp.potential_from_grid(grid=grid)).all()
        assert (
            sersic_mp.deflections_from_grid(grid=grid)
            == sersic_lmp.deflections_from_grid(grid=grid)
        ).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllipticalDevVaucouleurs(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
        )
        spherical = ag.lmp.SphericalDevVaucouleurs(
            centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
        )

        assert (
            elliptical.image_from_grid(grid=grid)
            == spherical.image_from_grid(grid=grid)
        ).all()
        assert (
            elliptical.convergence_from_grid(grid=grid)
            == spherical.convergence_from_grid(grid=grid)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )


class TestSersicRadialGradient:
    def test__constructor_and_units(self):
        sersic = ag.lmp.EllipticalSersicRadialGradient(
            centre=(1.0, 2.0),
            elliptical_comps=(0.333333, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
            mass_to_light_gradient=-1.0,
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

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, ag.dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == pytest.approx(
            0.6 / np.sqrt(0.5), 1.0e-4
        )

        sersic = ag.lmp.SphericalSersicRadialGradient(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
            mass_to_light_gradient=-1.0,
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

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, ag.dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__grid_calculations__same_as_sersic_radial_gradient(self):
        sersic_lp = ag.lmp.EllipticalSersic(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
        )
        sersic_mp = ag.lmp.EllipticalSersicRadialGradient(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=0.5,
        )
        sersic_lmp = ag.lmp.EllipticalSersicRadialGradient(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=0.5,
        )

        assert (
            sersic_lp.image_from_grid(grid=grid)
            == sersic_lmp.image_from_grid(grid=grid)
        ).all()
        assert (
            sersic_mp.convergence_from_grid(grid=grid)
            == sersic_lmp.convergence_from_grid(grid=grid)
        ).all()
        #    assert (sersic_mp.potential_from_grid(grid=grid) == sersic_lmp.potential_from_grid(grid=grid)).all()
        assert (
            sersic_mp.deflections_from_grid(grid=grid)
            == sersic_lmp.deflections_from_grid(grid=grid)
        ).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllipticalSersicRadialGradient(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
        )
        spherical = ag.lmp.SphericalSersicRadialGradient(
            centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
        )

        assert (
            elliptical.image_from_grid(grid=grid)
            == spherical.image_from_grid(grid=grid)
        ).all()
        assert (
            elliptical.convergence_from_grid(grid=grid)
            == spherical.convergence_from_grid(grid=grid)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )
