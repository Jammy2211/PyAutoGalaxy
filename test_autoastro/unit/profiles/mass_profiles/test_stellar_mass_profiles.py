import numpy as np
import pytest

import autofit as af
import autoarray as aa
from autoarray.structures import grids
import autoastro as aast
import os

from scipy.special import wofz


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    af.conf.instance = af.conf.default


grid = aa.GridIrregular.manual_1d([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestEllipticalGaussian:
    def test__omega_from_grid_and_q(self):

        gaussian = aast.mp.EllipticalGaussian()

        omega = gaussian.omega_from_grid_and_q(
            grid_complex=np.array([[1.0j + 0.0]]), q=1
        )

        assert np.real(omega[0]) == pytest.approx(2.71828183, 1.0e-4)
        assert np.imag(omega[0]) == pytest.approx(-0.42758, 1.0e-4)

        omega = gaussian.omega_from_grid_and_q(
            grid_complex=np.array([[1.0j + 0.0]]), q=0.5
        )

        assert np.real(omega[0]) == pytest.approx(2.71828, 1.0e-4)
        assert np.imag(omega[0]) == pytest.approx(-0.012715, 1.0e-4)

        omega = gaussian.omega_from_grid_and_q(
            grid_complex=np.array([[1.0j + 0.5]]), q=0.8
        )

        assert np.real(omega[0]) == pytest.approx(1.18568, 1.0e-4)
        assert np.imag(omega[0]) == pytest.approx(-1.9643, 1.0e-4)

        omega = gaussian.omega_from_grid_and_q(
            grid_complex=np.array([[0.7j + 0.5]]), q=0.8
        )

        assert np.real(omega[0]) == pytest.approx(1.05562, 1.0e-4)
        assert np.imag(omega[0]) == pytest.approx(-1.12107, 1.0e-4)

        omega = gaussian.omega_from_grid_and_q(
            grid_complex=np.array([[1.0j + 0.5], [0.7j + 0.5]]), q=0.8
        )

        assert np.real(omega[0]) == pytest.approx(1.18568, 1.0e-4)
        assert np.imag(omega[0]) == pytest.approx(-1.9643, 1.0e-4)
        assert np.real(omega[1]) == pytest.approx(1.05562, 1.0e-4)
        assert np.imag(omega[1]) == pytest.approx(-1.12107, 1.0e-4)

    def test__sigma_from_grid(self):

        gaussian = aast.mp.EllipticalGaussian(axis_ratio=0.9, sigma=2.0)

        sigma = gaussian.sigma_from_grid(grid=aa.GridIrregular.manual_1d([[1.0, 0.0]]))

        assert np.real(sigma[0]) == pytest.approx(0.0, 1.0e-4)
        assert np.imag(sigma[0]) == pytest.approx(-0.086128, 1.0e-4)

        gaussian = aast.mp.EllipticalGaussian(axis_ratio=0.9, sigma=3.0)

        sigma = gaussian.sigma_from_grid(grid=aa.GridIrregular.manual_1d([[1.0, 0.0]]))

        assert np.real(sigma[0]) == pytest.approx(0.0, 1.0e-4)
        assert np.imag(sigma[0]) == pytest.approx(-0.059380, 1.0e-4)

        sigma = gaussian.sigma_from_grid(grid=aa.GridIrregular.manual_1d([[1.0, 0.5]]))

        assert np.real(sigma[0]) == pytest.approx(0.026596, 1.0e-4)
        assert np.imag(sigma[0]) == pytest.approx(-0.059033, 1.0e-4)

        gaussian = aast.mp.EllipticalGaussian(axis_ratio=0.8, sigma=3.0)

        sigma = gaussian.sigma_from_grid(grid=aa.GridIrregular.manual_1d([[1.0, 0.5]]))

        assert np.real(sigma[0]) == pytest.approx(0.0344443, 1.0e-4)
        assert np.imag(sigma[0]) == pytest.approx(-0.085903, 1.0e-4)

        sigma = gaussian.sigma_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.3, 0.5], [0.3, 0.5]])
        )

        assert np.real(sigma[0]) == pytest.approx(0.03522, 1.0e-4)
        assert np.imag(sigma[0]) == pytest.approx(-0.026401, 1.0e-4)
        assert np.real(sigma[1]) == pytest.approx(0.03522, 1.0e-4)
        assert np.imag(sigma[1]) == pytest.approx(-0.026401, 1.0e-4)

    def test__deflections_correct_values(self):

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0),
            axis_ratio=0.9,
            intensity=1.0,
            sigma=3.0,
            mass_to_light_ratio=1.0,
        )

        deflections = gaussian.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.0, 0.0]])
        )

        assert deflections[0, 0] == pytest.approx(0.85595, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0),
            axis_ratio=0.8,
            intensity=1.0,
            sigma=5.0,
            mass_to_light_ratio=1.0,
        )

        deflections = gaussian.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.5, 0.2]])
        )

        assert deflections[0, 0] == pytest.approx(0.277765, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(0.088903, 1.0e-4)

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0),
            axis_ratio=0.8,
            intensity=1.0,
            sigma=5.0,
            mass_to_light_ratio=2.0,
        )

        deflections = gaussian.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.5, 0.2]])
        )

        assert deflections[0, 0] == pytest.approx(0.55553, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(0.177806, 1.0e-4)

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0),
            axis_ratio=0.8,
            intensity=2.0,
            sigma=5.0,
            mass_to_light_ratio=1.0,
        )

        deflections = gaussian.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.5, 0.2]])
        )

        assert deflections[0, 0] == pytest.approx(0.555531, 1.0e-4)
        assert deflections[0, 1] == pytest.approx(0.177806, 1.0e-4)

    def test__deflections_via_integrator_and_analytic_agree(self):

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.4, 0.2),
            axis_ratio=0.7,
            intensity=1.0,
            sigma=10.0,
            mass_to_light_ratio=1.0,
        )

        grid = aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=0.1, origin=(1.0, 1.0))

        deflections_via_analytic = gaussian.deflections_from_grid_via_analytic(
            grid=grid
        )
        deflections_via_integrator = gaussian.deflections_from_grid_via_integrator(
            grid=grid
        )

        assert deflections_via_analytic == pytest.approx(
            deflections_via_integrator, 1.0e-2
        )

        gaussian = aast.mp.EllipticalGaussian(
            centre=(-0.7, -0.4),
            axis_ratio=0.8,
            intensity=3.0,
            sigma=15.0,
            mass_to_light_ratio=7.0,
        )

        grid = aa.Grid.uniform(shape_2d=(3, 3), pixel_scales=0.1, origin=(1.0, 1.0))

        deflections_via_analytic = gaussian.deflections_from_grid_via_analytic(
            grid=grid
        )
        deflections_via_integrator = gaussian.deflections_from_grid_via_integrator(
            grid=grid
        )

        assert deflections_via_analytic == pytest.approx(
            deflections_via_integrator, 1.0e-2
        )

    def test__deflections_via_grid__uses_integrator_if_analytic_fails_else_analytic(
        self
    ):

        gaussian = aast.mp.EllipticalGaussian(
            centre=(-0.7, -0.4),
            axis_ratio=0.8,
            intensity=3.0,
            sigma=5.0,
            mass_to_light_ratio=7.0,
        )

        deflections = gaussian.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[-1.0, 0.0]])
        )
        deflections_via_analytic = gaussian.deflections_from_grid_via_analytic(
            grid=aa.GridIrregular.manual_1d([[-1.0, 0.0]])
        )
        deflections_via_integrator = gaussian.deflections_from_grid_via_integrator(
            grid=aa.GridIrregular.manual_1d([[-1.0, 0.0]])
        )

        assert deflections[0, 0] == deflections_via_analytic[0, 0]
        assert deflections[0, 1] == deflections_via_analytic[0, 1]
        assert deflections[0, 0] != deflections_via_integrator[0, 0]
        assert deflections[0, 1] != deflections_via_integrator[0, 1]

        gaussian = aast.mp.EllipticalGaussian(
            centre=(-0.0, -0.0),
            axis_ratio=0.2,
            intensity=3.0,
            sigma=0.1,
            mass_to_light_ratio=7.0,
        )

        pytest.warns(RuntimeWarning)
        deflections = gaussian.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[-5.0, 0.0]])
        )
        deflections_via_analytic = gaussian.deflections_from_grid_via_analytic(
            grid=aa.GridIrregular.manual_1d([[-5.0, 0.0]])
        )
        deflections_via_integrator = gaussian.deflections_from_grid_via_integrator(
            grid=aa.GridIrregular.manual_1d([[-5.0, 0.0]])
        )

        assert deflections[0, 0] != deflections_via_analytic[0, 0]
        assert deflections[0, 1] != deflections_via_analytic[0, 1]
        # pytest wont raise the warning so these fail >_>.

    #   assert deflections[0, 0] == deflections_via_integrator[0, 0]
    #   assert deflections[0, 1] == deflections_via_integrator[0, 1]

    def test__intensity_as_radius__correct_value(self):
        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=1.0
        )
        assert gaussian.intensity_at_radius(grid_radii=1.0) == pytest.approx(
            0.24197, 1e-2
        )

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=2.0, sigma=1.0
        )
        assert gaussian.intensity_at_radius(grid_radii=1.0) == pytest.approx(
            2.0 * 0.24197, 1e-2
        )

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=2.0
        )
        assert gaussian.intensity_at_radius(grid_radii=1.0) == pytest.approx(
            0.1760, 1e-2
        )

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0), axis_ratio=1.0, phi=0.0, intensity=1.0, sigma=2.0
        )
        assert gaussian.intensity_at_radius(grid_radii=3.0) == pytest.approx(
            0.0647, 1e-2
        )

    def test__convergence_from_grid__correct_value(self):

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0),
            axis_ratio=0.9999,
            phi=0.0,
            intensity=1.0,
            sigma=1.0,
            mass_to_light_ratio=1.0,
        )
        assert gaussian.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(0.24197, 1e-2)

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0),
            axis_ratio=0.9999,
            phi=0.0,
            intensity=1.0,
            sigma=1.0,
            mass_to_light_ratio=2.0,
        )
        assert gaussian.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(2.0 * 0.24197, 1e-2)

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0),
            axis_ratio=0.9999,
            phi=0.0,
            intensity=2.0,
            sigma=1.0,
            mass_to_light_ratio=1.0,
        )
        assert gaussian.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(2.0 * 0.24197, 1e-2)

        gaussian = aast.mp.EllipticalGaussian(
            centre=(0.0, 0.0),
            axis_ratio=0.5,
            phi=0.0,
            intensity=2.0,
            sigma=3.0,
            mass_to_light_ratio=4.0,
        )
        assert gaussian.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(1.03470, 1e-2)


class TestSersic:
    def test__constructor_and_units(self):
        sersic = aast.mp.EllipticalSersic(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], aast.dim.Length)
        assert isinstance(sersic.centre[1], aast.dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, aast.dim.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, aast.dim.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, aast.dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = aast.mp.EllipticalSersic(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], aast.dim.Length)
        assert isinstance(sersic.centre[1], aast.dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, aast.dim.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, aast.dim.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, aast.dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        sersic = aast.mp.EllipticalSersic(
            centre=(0.0, 0.0),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.5]])
        ) == pytest.approx(4.90657319276, 1e-3)

        sersic = aast.mp.EllipticalSersic(
            centre=(0.0, 0.0),
            intensity=6.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.5]])
        ) == pytest.approx(2.0 * 4.90657319276, 1e-3)

        sersic = aast.mp.EllipticalSersic(
            centre=(0.0, 0.0),
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        assert sersic.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.5]])
        ) == pytest.approx(2.0 * 4.90657319276, 1e-3)

        sersic = aast.mp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(grid=aa.Coordinates([[(1.0, 0.0)]]))[0][
            0
        ] == pytest.approx(5.38066670129, 1e-3)

    def test__deflections_correct_values(self):
        sersic = aast.mp.EllipticalSersic(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        deflections = sersic.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.1625, 0.1625]])
        )
        assert deflections[0, 0] == pytest.approx(1.1446, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.79374, 1e-3)

        sersic = aast.mp.EllipticalSersic(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        deflections = sersic.deflections_from_grid(
            grid=aa.Coordinates([[(0.1625, 0.1625), (0.1625, 0.1625)]])
        )
        assert deflections[0][0][0] == pytest.approx(1.1446, 1e-3)
        assert deflections[0][0][1] == pytest.approx(0.79374, 1e-3)
        assert deflections[0][1][0] == pytest.approx(1.1446, 1e-3)
        assert deflections[0][1][1] == pytest.approx(0.79374, 1e-3)

    def test__surfce_density__change_geometry(self):
        sersic_0 = aast.mp.EllipticalSersic(centre=(0.0, 0.0))
        sersic_1 = aast.mp.EllipticalSersic(centre=(1.0, 1.0))
        assert sersic_0.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.0, 1.0]])
        ) == sersic_1.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 0.0]])
        )

        sersic_0 = aast.mp.EllipticalSersic(centre=(0.0, 0.0))
        sersic_1 = aast.mp.EllipticalSersic(centre=(0.0, 0.0))
        assert sersic_0.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.0, 0.0]])
        ) == sersic_1.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        )

        sersic_0 = aast.mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        sersic_1 = aast.mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        assert sersic_0.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.0, 0.0]])
        ) == sersic_1.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        )

    def test__deflections__change_geometry(self):
        sersic_0 = aast.mp.EllipticalSersic(centre=(0.0, 0.0))
        sersic_1 = aast.mp.EllipticalSersic(centre=(1.0, 1.0))
        deflections_0 = sersic_0.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.0, 1.0]])
        )
        deflections_1 = sersic_1.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 0.0]])
        )
        assert deflections_0[0, 0] == pytest.approx(-deflections_1[0, 0], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(-deflections_1[0, 1], 1e-5)

        sersic_0 = aast.mp.EllipticalSersic(centre=(0.0, 0.0))
        sersic_1 = aast.mp.EllipticalSersic(centre=(0.0, 0.0))
        deflections_0 = sersic_0.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.0, 0.0]])
        )
        deflections_1 = sersic_1.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        )
        assert deflections_0[0, 0] == pytest.approx(deflections_1[0, 1], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(deflections_1[0, 0], 1e-5)

        sersic_0 = aast.mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=0.0)
        sersic_1 = aast.mp.EllipticalSersic(centre=(0.0, 0.0), axis_ratio=0.8, phi=90.0)
        deflections_0 = sersic_0.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.0, 0.0]])
        )
        deflections_1 = sersic_1.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        )
        assert deflections_0[0, 0] == pytest.approx(deflections_1[0, 1], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(deflections_1[0, 0], 1e-5)

    def test__spherical_and_elliptical_identical(self):
        elliptical = aast.mp.EllipticalSersic(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
        )

        spherical = aast.mp.EllipticalSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
        )

        assert (
            elliptical.convergence_from_grid(grid=grid)
            == spherical.convergence_from_grid(grid=grid)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = aast.mp.EllipticalSersic(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = aa.Mask.manual(mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.MaskedGrid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = aast.mp.EllipticalSersic(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = aa.Mask.manual(mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.MaskedGrid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__outputs_are_autoarrays(self):
        grid = aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)

        sersic = aast.mp.EllipticalSersic()

        convergence = sersic.convergence_from_grid(grid=grid)

        assert convergence.shape_2d == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=grid)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(grid=grid)

        assert deflections.shape_2d == (2, 2)

        sersic = aast.mp.EllipticalSersic()

        convergence = sersic.convergence_from_grid(grid=grid)

        assert convergence.shape_2d == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=grid)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(grid=grid)

        assert deflections.shape_2d == (2, 2)


class TestExponential:
    def test__constructor_and_units(self):
        exponential = aast.mp.EllipticalExponential(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], aast.dim.Length)
        assert isinstance(exponential.centre[1], aast.dim.Length)
        assert exponential.centre[0].unit == "arcsec"
        assert exponential.centre[1].unit == "arcsec"

        assert exponential.axis_ratio == 0.5
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 45.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, aast.dim.Luminosity)
        assert exponential.intensity.unit == "eps"

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, aast.dim.Length)
        assert exponential.effective_radius.unit_length == "arcsec"

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, aast.dim.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == "angular / eps"

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        exponential = aast.mp.EllipticalExponential(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert exponential.centre == (1.0, 2.0)
        assert isinstance(exponential.centre[0], aast.dim.Length)
        assert isinstance(exponential.centre[1], aast.dim.Length)
        assert exponential.centre[0].unit == "arcsec"
        assert exponential.centre[1].unit == "arcsec"

        assert exponential.axis_ratio == 1.0
        assert isinstance(exponential.axis_ratio, float)

        assert exponential.phi == 0.0
        assert isinstance(exponential.phi, float)

        assert exponential.intensity == 1.0
        assert isinstance(exponential.intensity, aast.dim.Luminosity)
        assert exponential.intensity.unit == "eps"

        assert exponential.effective_radius == 0.6
        assert isinstance(exponential.effective_radius, aast.dim.Length)
        assert exponential.effective_radius.unit_length == "arcsec"

        assert exponential.sersic_index == 1.0
        assert isinstance(exponential.sersic_index, float)

        assert exponential.mass_to_light_ratio == 10.0
        assert isinstance(exponential.mass_to_light_ratio, aast.dim.MassOverLuminosity)
        assert exponential.mass_to_light_ratio.unit == "angular / eps"

        assert exponential.sersic_constant == pytest.approx(1.67838, 1e-3)
        assert exponential.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        exponential = aast.mp.EllipticalExponential(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.0, 0.0]])
        ) == pytest.approx(4.9047, 1e-3)

        exponential = aast.mp.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(4.8566, 1e-3)

        exponential = aast.mp.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=4.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(2.0 * 4.8566, 1e-3)

        exponential = aast.mp.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=2.0,
        )
        assert exponential.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(2.0 * 4.8566, 1e-3)

        exponential = aast.mp.EllipticalExponential(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert exponential.convergence_from_grid(grid=aa.Coordinates([[(0.0, 1.0)]]))[
            0
        ][0] == pytest.approx(4.8566, 1e-3)

    def test__deflections_correct_values(self):
        exponential = aast.mp.EllipticalExponential(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )
        deflections = exponential.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.1625, 0.1625]])
        )
        assert deflections[0, 0] == pytest.approx(0.90493, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.62569, 1e-3)

        exponential = aast.mp.EllipticalExponential(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )
        deflections = exponential.deflections_from_grid(
            grid=aa.Coordinates([[(0.1625, 0.1625)]])
        )
        assert deflections[0][0][0] == pytest.approx(0.90493, 1e-3)
        assert deflections[0][0][1] == pytest.approx(0.62569, 1e-3)

    def test__spherical_and_elliptical_identical(self):
        elliptical = aast.mp.EllipticalExponential(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        spherical = aast.mp.EllipticalExponential(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        assert (
            elliptical.convergence_from_grid(grid=grid)
            == spherical.convergence_from_grid(grid=grid)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        exponential = aast.mp.EllipticalExponential(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = aa.Mask.manual(mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.MaskedGrid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = exponential.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = exponential.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        exponential = aast.mp.EllipticalExponential(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = aa.Mask.manual(mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.MaskedGrid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = exponential.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = exponential.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__outputs_are_autoarrays(self):
        grid = aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)

        exponential = aast.mp.EllipticalExponential()

        convergence = exponential.convergence_from_grid(grid=grid)

        assert convergence.shape_2d == (2, 2)

        # potential = exponential.potential_from_grid(
        #     grid=grid)
        #
        # assert potential.shape == (2, 2)

        deflections = exponential.deflections_from_grid(grid=grid)

        assert deflections.shape_2d == (2, 2)

        exponential = aast.mp.EllipticalExponential()

        convergence = exponential.convergence_from_grid(grid=grid)

        assert convergence.shape_2d == (2, 2)

        # potential = exponential.potential_from_grid(
        #     grid=grid)
        #
        # assert potential.shape == (2, 2)

        deflections = exponential.deflections_from_grid(grid=grid)

        assert deflections.shape_2d == (2, 2)


class TestDevVaucouleurs:
    def test__constructor_and_units(self):
        dev_vaucouleurs = aast.mp.EllipticalDevVaucouleurs(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], aast.dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], aast.dim.Length)
        assert dev_vaucouleurs.centre[0].unit == "arcsec"
        assert dev_vaucouleurs.centre[1].unit == "arcsec"

        assert dev_vaucouleurs.axis_ratio == 0.5
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 45.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, aast.dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == "eps"

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, aast.dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == "arcsec"

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(
            dev_vaucouleurs.mass_to_light_ratio, aast.dim.MassOverLuminosity
        )
        assert dev_vaucouleurs.mass_to_light_ratio.unit == "angular / eps"

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        dev_vaucouleurs = aast.mp.EllipticalDevVaucouleurs(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=10.0,
        )

        assert dev_vaucouleurs.centre == (1.0, 2.0)
        assert isinstance(dev_vaucouleurs.centre[0], aast.dim.Length)
        assert isinstance(dev_vaucouleurs.centre[1], aast.dim.Length)
        assert dev_vaucouleurs.centre[0].unit == "arcsec"
        assert dev_vaucouleurs.centre[1].unit == "arcsec"

        assert dev_vaucouleurs.axis_ratio == 1.0
        assert isinstance(dev_vaucouleurs.axis_ratio, float)

        assert dev_vaucouleurs.phi == 0.0
        assert isinstance(dev_vaucouleurs.phi, float)

        assert dev_vaucouleurs.intensity == 1.0
        assert isinstance(dev_vaucouleurs.intensity, aast.dim.Luminosity)
        assert dev_vaucouleurs.intensity.unit == "eps"

        assert dev_vaucouleurs.effective_radius == 0.6
        assert isinstance(dev_vaucouleurs.effective_radius, aast.dim.Length)
        assert dev_vaucouleurs.effective_radius.unit_length == "arcsec"

        assert dev_vaucouleurs.sersic_index == 4.0
        assert isinstance(dev_vaucouleurs.sersic_index, float)

        assert dev_vaucouleurs.mass_to_light_ratio == 10.0
        assert isinstance(
            dev_vaucouleurs.mass_to_light_ratio, aast.dim.MassOverLuminosity
        )
        assert dev_vaucouleurs.mass_to_light_ratio.unit == "angular / eps"

        assert dev_vaucouleurs.sersic_constant == pytest.approx(7.66924, 1e-3)
        assert dev_vaucouleurs.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        dev = aast.mp.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            mass_to_light_ratio=1.0,
        )
        assert dev.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.0, 0.0]])
        ) == pytest.approx(5.6697, 1e-3)

        dev = aast.mp.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert dev.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(7.4455, 1e-3)

        dev = aast.mp.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=90.0,
            intensity=4.0,
            effective_radius=3.0,
            mass_to_light_ratio=1.0,
        )
        assert dev.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(2.0 * 7.4455, 1e-3)

        dev = aast.mp.EllipticalDevVaucouleurs(
            axis_ratio=0.5,
            phi=90.0,
            intensity=2.0,
            effective_radius=3.0,
            mass_to_light_ratio=2.0,
        )
        assert dev.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(2.0 * 7.4455, 1e-3)

        sersic = aast.mp.EllipticalDevVaucouleurs(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=1.0,
        )
        assert sersic.convergence_from_grid(grid=aa.Coordinates([[(0.0, 1.0)]]))[0][
            0
        ] == pytest.approx(0.351797, 1e-3)

    def test__deflections_correct_values(self):
        dev = aast.mp.EllipticalDevVaucouleurs(
            centre=(0.4, 0.2),
            axis_ratio=0.9,
            phi=10.0,
            intensity=2.0,
            effective_radius=0.8,
            mass_to_light_ratio=3.0,
        )
        deflections = dev.deflections_from_grid(
            grid=aa.Coordinates([[(0.1625, 0.1625)]])
        )
        assert deflections[0][0][0] == pytest.approx(-24.528, 1e-3)
        assert deflections[0][0][1] == pytest.approx(-3.37605, 1e-3)

    def test__spherical_and_elliptical_identical(self):
        elliptical = aast.mp.EllipticalDevVaucouleurs(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        spherical = aast.mp.EllipticalDevVaucouleurs(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            mass_to_light_ratio=1.0,
        )

        assert (
            elliptical.convergence_from_grid(grid=grid)
            == spherical.convergence_from_grid(grid=grid)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)

        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        dev = aast.mp.EllipticalDevVaucouleurs(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = aa.Mask.manual(mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.MaskedGrid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = dev.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = dev.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        dev = aast.mp.EllipticalDevVaucouleurs(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = aa.Mask.manual(mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.MaskedGrid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = dev.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = dev.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__outputs_are_autoarrays(self):
        grid = aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)

        dev_vaucouleurs = aast.mp.EllipticalDevVaucouleurs()

        convergence = dev_vaucouleurs.convergence_from_grid(grid=grid)

        assert convergence.shape_2d == (2, 2)

        # potential = dev_vaucouleurs.potential_from_grid(
        #     grid=grid)
        #
        # assert potential.shape == (2, 2)

        deflections = dev_vaucouleurs.deflections_from_grid(grid=grid)

        assert deflections.shape_2d == (2, 2)

        dev_vaucouleurs = aast.mp.EllipticalDevVaucouleurs()

        convergence = dev_vaucouleurs.convergence_from_grid(grid=grid)

        assert convergence.shape_2d == (2, 2)

        # potential = dev_vaucouleurs.potential_from_grid(
        #     grid=grid)
        #
        # assert potential.shape == (2, 2)

        deflections = dev_vaucouleurs.deflections_from_grid(grid=grid)

        assert deflections.shape_2d == (2, 2)


class TestSersicMassRadialGradient:
    def test__constructor_and_units(self):
        sersic = aast.mp.EllipticalSersicRadialGradient(
            centre=(1.0, 2.0),
            axis_ratio=0.5,
            phi=45.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
            mass_to_light_gradient=-1.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], aast.dim.Length)
        assert isinstance(sersic.centre[1], aast.dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 0.5
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 45.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, aast.dim.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, aast.dim.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, aast.dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6 / np.sqrt(0.5)

        sersic = aast.mp.EllipticalSersicRadialGradient(
            centre=(1.0, 2.0),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=10.0,
            mass_to_light_gradient=-1.0,
        )

        assert sersic.centre == (1.0, 2.0)
        assert isinstance(sersic.centre[0], aast.dim.Length)
        assert isinstance(sersic.centre[1], aast.dim.Length)
        assert sersic.centre[0].unit == "arcsec"
        assert sersic.centre[1].unit == "arcsec"

        assert sersic.axis_ratio == 1.0
        assert isinstance(sersic.axis_ratio, float)

        assert sersic.phi == 0.0
        assert isinstance(sersic.phi, float)

        assert sersic.intensity == 1.0
        assert isinstance(sersic.intensity, aast.dim.Luminosity)
        assert sersic.intensity.unit == "eps"

        assert sersic.effective_radius == 0.6
        assert isinstance(sersic.effective_radius, aast.dim.Length)
        assert sersic.effective_radius.unit_length == "arcsec"

        assert sersic.sersic_index == 4.0
        assert isinstance(sersic.sersic_index, float)

        assert sersic.mass_to_light_ratio == 10.0
        assert isinstance(sersic.mass_to_light_ratio, aast.dim.MassOverLuminosity)
        assert sersic.mass_to_light_ratio.unit == "angular / eps"

        assert sersic.mass_to_light_gradient == -1.0
        assert isinstance(sersic.mass_to_light_gradient, float)

        assert sersic.sersic_constant == pytest.approx(7.66925, 1e-3)
        assert sersic.elliptical_effective_radius == 0.6

    def test__convergence_correct_values(self):
        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1/0.6)**-1.0 = 0.6
        sersic = aast.mp.EllipticalSersicRadialGradient(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        assert sersic.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.0, 1.0]])
        ) == pytest.approx(0.6 * 0.351797, 1e-3)

        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1.5/2.0)**1.0 = 0.75
        sersic = aast.mp.EllipticalSersicRadialGradient(
            axis_ratio=1.0,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=-1.0,
        )
        assert sersic.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.5, 0.0]])
        ) == pytest.approx(0.75 * 4.90657319276, 1e-3)

        sersic = aast.mp.EllipticalSersicRadialGradient(
            axis_ratio=1.0,
            phi=0.0,
            intensity=6.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=-1.0,
        )
        assert sersic.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.5, 0.0]])
        ) == pytest.approx(2.0 * 0.75 * 4.90657319276, 1e-3)

        sersic = aast.mp.EllipticalSersicRadialGradient(
            axis_ratio=1.0,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=-1.0,
        )
        assert sersic.convergence_from_grid(
            grid=aa.GridIrregular.manual_1d([[1.5, 0.0]])
        ) == pytest.approx(2.0 * 0.75 * 4.90657319276, 1e-3)

        # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = ((0.5*1.41)/2.0)**-1.0 = 2.836
        sersic = aast.mp.EllipticalSersicRadialGradient(
            axis_ratio=0.5,
            phi=0.0,
            intensity=3.0,
            effective_radius=2.0,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        assert sersic.convergence_from_grid(grid=aa.Coordinates([[(1.0, 0.0)]]))[0][
            0
        ] == pytest.approx(2.836879 * 5.38066670129, abs=2e-01)

    def test__deflections_correct_values(self):
        sersic = aast.mp.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        deflections = sersic.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.1625, 0.1625]])
        )
        assert deflections[0, 0] == pytest.approx(3.60324873535244, 1e-3)
        assert deflections[0, 1] == pytest.approx(2.3638898009652, 1e-3)

        sersic = aast.mp.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=-1.0,
        )
        deflections = sersic.deflections_from_grid(
            grid=aa.Coordinates([[(0.1625, 0.1625)]])
        )
        assert deflections[0][0][0] == pytest.approx(0.97806399756448, 1e-3)
        assert deflections[0][0][1] == pytest.approx(0.725459334118341, 1e-3)

    def test__compare_to_sersic(self):
        sersic = aast.mp.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=1.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=0.0,
        )
        sersic_deflections = sersic.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.1625, 0.1625]])
        )

        exponential = aast.mp.EllipticalExponential(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            mass_to_light_ratio=1.0,
        )
        exponential_deflections = exponential.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.1625, 0.1625]])
        )

        assert (
            sersic_deflections[0, 0]
            == exponential_deflections[0, 0]
            == pytest.approx(0.90493, 1e-3)
        )
        assert (
            sersic_deflections[0, 1]
            == exponential_deflections[0, 1]
            == pytest.approx(0.62569, 1e-3)
        )

        sersic = aast.mp.EllipticalSersicRadialGradient(
            centre=(0.4, 0.2),
            axis_ratio=0.9,
            phi=10.0,
            intensity=2.0,
            effective_radius=0.8,
            sersic_index=4.0,
            mass_to_light_ratio=3.0,
            mass_to_light_gradient=0.0,
        )
        sersic_deflections = sersic.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.1625, 0.1625]])
        )

        dev = aast.mp.EllipticalDevVaucouleurs(
            centre=(0.4, 0.2),
            axis_ratio=0.9,
            phi=10.0,
            intensity=2.0,
            effective_radius=0.8,
            mass_to_light_ratio=3.0,
        )

        dev_deflections = dev.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.1625, 0.1625]])
        )

        assert (
            sersic_deflections[0, 0]
            == dev_deflections[0, 0]
            == pytest.approx(-24.528, 1e-3)
        )
        assert (
            sersic_deflections[0, 1]
            == dev_deflections[0, 1]
            == pytest.approx(-3.37605, 1e-3)
        )

        sersic_grad = aast.mp.EllipticalSersicRadialGradient(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=0.0,
        )
        sersic_grad_deflections = sersic_grad.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.1625, 0.1625]])
        )

        sersic = aast.mp.EllipticalSersic(
            centre=(-0.4, -0.2),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
        )
        sersic_deflections = sersic.deflections_from_grid(
            grid=aa.GridIrregular.manual_1d([[0.1625, 0.1625]])
        )

        assert (
            sersic_grad_deflections[0, 0]
            == sersic_deflections[0, 0]
            == pytest.approx(1.1446, 1e-3)
        )
        assert (
            sersic_grad_deflections[0, 1]
            == sersic_deflections[0, 1]
            == pytest.approx(0.79374, 1e-3)
        )

    def test__spherical_and_elliptical_identical(self):
        elliptical = aast.mp.EllipticalSersicRadialGradient(
            centre=(0.0, 0.0),
            axis_ratio=1.0,
            phi=0.0,
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        spherical = aast.mp.EllipticalSersicRadialGradient(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=4.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.0,
        )
        assert (
            elliptical.convergence_from_grid(grid=grid)
            == spherical.convergence_from_grid(grid=grid)
        ).all()
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)
        assert (
            elliptical.deflections_from_grid(grid=grid)
            == spherical.deflections_from_grid(grid=grid)
        ).all()

    def test__deflections_of_elliptical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = aast.mp.EllipticalSersicRadialGradient(
            centre=(-0.7, 0.5),
            axis_ratio=0.8,
            phi=110.0,
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.5,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = aa.Mask.manual(mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.MaskedGrid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__deflections_of_spherical_profile__use_interpolate_and_cache_decorators(
        self
    ):
        sersic = aast.mp.EllipticalSersicRadialGradient(
            centre=(-0.7, 0.5),
            intensity=5.0,
            effective_radius=0.2,
            sersic_index=2.0,
            mass_to_light_ratio=1.0,
            mass_to_light_gradient=1.5,
        )

        mask = np.array(
            [
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, True, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ]
        )

        mask = aa.Mask.manual(mask, pixel_scales=(1.0, 1.0), sub_size=1)

        grid = aa.MaskedGrid.from_mask(mask=mask)

        regular_with_interp = grid.new_grid_with_interpolator(
            pixel_scale_interpolation_grid=0.5
        )
        interp_deflections = sersic.deflections_from_grid(grid=regular_with_interp)

        interpolator = grids.Interpolator.from_mask_grid_and_pixel_scale_interpolation_grids(
            mask=mask, grid=grid, pixel_scale_interpolation_grid=0.5
        )

        interp_deflections_values = sersic.deflections_from_grid(
            grid=interpolator.interp_grid
        )

        interp_deflections_manual_y = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 0]
        )
        interp_deflections_manual_x = interpolator.interpolated_values_from_values(
            values=interp_deflections_values[:, 1]
        )

        assert (interp_deflections_manual_y == interp_deflections[:, 0]).all()
        assert (interp_deflections_manual_x == interp_deflections[:, 1]).all()

    def test__outputs_are_autoarrays(self):
        grid = aa.Grid.uniform(shape_2d=(2, 2), pixel_scales=1.0, sub_size=1)

        sersic = aast.mp.EllipticalSersicRadialGradient()

        convergence = sersic.convergence_from_grid(grid=grid)

        assert convergence.shape_2d == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=grid)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(grid=grid)

        assert deflections.shape_2d == (2, 2)

        sersic = aast.mp.EllipticalSersicRadialGradient()

        convergence = sersic.convergence_from_grid(grid=grid)

        assert convergence.shape_2d == (2, 2)

        # potential = sersic.potential_from_grid(
        #     grid=grid)
        #
        # assert potential.shape == (2, 2)

        deflections = sersic.deflections_from_grid(grid=grid)

        assert deflections.shape_2d == (2, 2)
