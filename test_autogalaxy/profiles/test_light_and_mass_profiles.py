import autogalaxy as ag
import numpy as np
import pytest

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestGaussian(object):
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

        assert elliptical.image_from_grid(grid=grid) == pytest.approx(
            spherical.image_from_grid(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_from_grid(grid=grid) == pytest.approx(
            spherical.convergence_from_grid(grid=grid), 1.0e-4
        )
        # assert (elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)).all()
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )


class TestExponential:
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

        assert elliptical.image_from_grid(grid=grid) == pytest.approx(
            spherical.image_from_grid(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_from_grid(grid=grid) == pytest.approx(
            spherical.convergence_from_grid(grid=grid), 1.0e-4
        )
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )


class TestDevVaucouleurs:
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

        assert elliptical.image_from_grid(grid=grid) == pytest.approx(
            spherical.image_from_grid(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_from_grid(grid=grid) == pytest.approx(
            spherical.convergence_from_grid(grid=grid), 1.0e-4
        )
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )


class TestSersicRadialGradient:
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

        assert sersic_lp.image_from_grid(grid=grid) == pytest.approx(
            sersic_lmp.image_from_grid(grid=grid), 1.0e-4
        )
        assert sersic_mp.convergence_from_grid(grid=grid) == pytest.approx(
            sersic_lmp.convergence_from_grid(grid=grid), 1.0e-4
        )
        #    assert (sersic_mp.potential_from_grid(grid=grid) == sersic_lmp.potential_from_grid(grid=grid)).all()
        assert sersic_mp.deflections_from_grid(grid=grid) == pytest.approx(
            sersic_lmp.deflections_from_grid(grid=grid), 1.0e-4
        )

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

        assert elliptical.image_from_grid(grid=grid) == pytest.approx(
            spherical.image_from_grid(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_from_grid(grid=grid) == pytest.approx(
            spherical.convergence_from_grid(grid=grid), 1.0e-4
        )
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )


class TestExponentialRadialGradient:
    def test__grid_calculations__same_as_sersic_radial_gradient(self):
        sersic_lp = ag.lmp.EllipticalExponential(
            elliptical_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6
        )
        sersic_mp = ag.lmp.EllipticalExponentialRadialGradient(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=0.5,
        )
        sersic_lmp = ag.lmp.EllipticalExponentialRadialGradient(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=0.5,
        )

        assert sersic_lp.image_from_grid(grid=grid) == pytest.approx(
            sersic_lmp.image_from_grid(grid=grid), 1.0e-4
        )
        assert sersic_mp.convergence_from_grid(grid=grid) == pytest.approx(
            sersic_lmp.convergence_from_grid(grid=grid), 1.0e-4
        )
        #    assert (sersic_mp.potential_from_grid(grid=grid) == sersic_lmp.potential_from_grid(grid=grid)).all()
        assert sersic_mp.deflections_from_grid(grid=grid) == pytest.approx(
            sersic_lmp.deflections_from_grid(grid=grid), 1.0e-4
        )

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllipticalExponentialRadialGradient(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
        )
        spherical = ag.lmp.SphericalExponentialRadialGradient(
            centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
        )

        assert elliptical.image_from_grid(grid=grid) == pytest.approx(
            spherical.image_from_grid(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_from_grid(grid=grid) == pytest.approx(
            spherical.convergence_from_grid(grid=grid), 1.0e-4
        )
        # assert elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )


class TestCoreSersic:
    def test__grid_calculations__same_as_core_sersic(self):
        sersic_lp = ag.lmp.EllipticalCoreSersic(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
        )
        sersic_mp = ag.lmp.EllipticalCoreSersic(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        sersic_lmp = ag.lmp.EllipticalCoreSersic(
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
        elliptical = ag.lmp.EllipticalCoreSersic(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        spherical = ag.lmp.SphericalCoreSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )

        assert elliptical.image_from_grid(grid=grid) == pytest.approx(
            spherical.image_from_grid(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_from_grid(grid=grid) == pytest.approx(
            spherical.convergence_from_grid(grid=grid), 1.0e-4
        )
        # assert (elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)).all()
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )


class TestChameleon:
    def test__grid_calculations__same_as_chameleon(self):
        chameleon_lp = ag.lmp.EllipticalChameleon(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
        )
        chameleon_mp = ag.lmp.EllipticalChameleon(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
            mass_to_light_ratio=2.0,
        )
        chameleon_lmp = ag.lmp.EllipticalChameleon(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
            mass_to_light_ratio=2.0,
        )

        assert (
            chameleon_lp.image_from_grid(grid=grid)
            == chameleon_lmp.image_from_grid(grid=grid)
        ).all()
        assert (
            chameleon_mp.convergence_from_grid(grid=grid)
            == chameleon_lmp.convergence_from_grid(grid=grid)
        ).all()
        #    assert (chameleon_mp.potential_from_grid(grid=grid) == chameleon_lmp.potential_from_grid(grid=grid)).all()
        assert (
            chameleon_mp.deflections_from_grid(grid=grid)
            == chameleon_lmp.deflections_from_grid(grid=grid)
        ).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllipticalChameleon(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
            mass_to_light_ratio=2.0,
        )
        spherical = ag.lmp.SphericalChameleon(
            centre=(0.0, 0.0),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
            mass_to_light_ratio=2.0,
        )

        assert elliptical.image_from_grid(grid=grid) == pytest.approx(
            spherical.image_from_grid(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_from_grid(grid=grid) == pytest.approx(
            spherical.convergence_from_grid(grid=grid), 1.0e4
        )
        # assert (elliptical.potential_from_grid(grid=grid) == spherical.potential_from_grid(grid=grid)).all()
        np.testing.assert_almost_equal(
            elliptical.deflections_from_grid(grid=grid),
            spherical.deflections_from_grid(grid=grid),
        )
