import autogalaxy as ag
import numpy as np
import pytest

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestGaussian(object):
    def test__grid_calculations__same_as_gaussian(self):

        gaussian_lp = ag.lmp.EllGaussian(
            elliptical_comps=(0.1, 0.05), intensity=1.0, sigma=5.0
        )
        gaussian_mp = ag.lmp.EllGaussian(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            sigma=5.0,
            mass_to_light_ratio=2.0,
        )
        gaussian_lmp = ag.lmp.EllGaussian(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            sigma=5.0,
            mass_to_light_ratio=2.0,
        )

        assert (
            gaussian_lp.image_2d_from(grid=grid)
            == gaussian_lmp.image_2d_from(grid=grid)
        ).all()
        assert (
            gaussian_mp.convergence_2d_from(grid=grid)
            == gaussian_lmp.convergence_2d_from(grid=grid)
        ).all()
        #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
        assert (
            gaussian_mp.deflections_yx_2d_from(grid=grid)
            == gaussian_lmp.deflections_yx_2d_from(grid=grid)
        ).all()


class TestSersic:
    def test__grid_calculations__same_as_sersic(self):

        sersic_lp = ag.lmp.EllSersic(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
        )
        sersic_mp = ag.lmp.EllSersic(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        sersic_lmp = ag.lmp.EllSersic(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )

        assert (
            sersic_lp.image_2d_from(grid=grid) == sersic_lmp.image_2d_from(grid=grid)
        ).all()
        assert (
            sersic_mp.convergence_2d_from(grid=grid)
            == sersic_lmp.convergence_2d_from(grid=grid)
        ).all()
        #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
        assert (
            sersic_mp.deflections_yx_2d_from(grid=grid)
            == sersic_lmp.deflections_yx_2d_from(grid=grid)
        ).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllSersic(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        spherical = ag.lmp.SphSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )

        assert elliptical.image_2d_from(grid=grid) == pytest.approx(
            spherical.image_2d_from(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
            spherical.convergence_2d_from(grid=grid), 1.0e-4
        )
        # assert (elliptical.potential_2d_from(grid=grid) == spherical.potential_2d_from(grid=grid)).all()
        np.testing.assert_almost_equal(
            elliptical.deflections_yx_2d_from(grid=grid),
            spherical.deflections_yx_2d_from(grid=grid),
        )


class TestExponential:
    def test__grid_calculations__same_as_exponential(self):

        sersic_lp = ag.lmp.EllExponential(
            elliptical_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6
        )
        sersic_mp = ag.lmp.EllExponential(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
        )
        sersic_lmp = ag.lmp.EllExponential(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
        )

        assert (
            sersic_lp.image_2d_from(grid=grid) == sersic_lmp.image_2d_from(grid=grid)
        ).all()
        assert (
            sersic_mp.convergence_2d_from(grid=grid)
            == sersic_lmp.convergence_2d_from(grid=grid)
        ).all()
        #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
        assert (
            sersic_mp.deflections_yx_2d_from(grid=grid)
            == sersic_lmp.deflections_yx_2d_from(grid=grid)
        ).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllExponential(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
        )
        spherical = ag.lmp.SphExponential(
            centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
        )

        assert elliptical.image_2d_from(grid=grid) == pytest.approx(
            spherical.image_2d_from(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
            spherical.convergence_2d_from(grid=grid), 1.0e-4
        )
        # assert elliptical.potential_2d_from(grid=grid) == spherical.potential_2d_from(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_yx_2d_from(grid=grid),
            spherical.deflections_yx_2d_from(grid=grid),
        )


class TestDevVaucouleurs:
    def test__grid_calculations__same_as_dev_vaucouleurs(self):

        sersic_lp = ag.lmp.EllDevVaucouleurs(
            elliptical_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6
        )
        sersic_mp = ag.lmp.EllDevVaucouleurs(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
        )
        sersic_lmp = ag.lmp.EllDevVaucouleurs(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
        )

        assert (
            sersic_lp.image_2d_from(grid=grid) == sersic_lmp.image_2d_from(grid=grid)
        ).all()
        assert (
            sersic_mp.convergence_2d_from(grid=grid)
            == sersic_lmp.convergence_2d_from(grid=grid)
        ).all()
        #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
        assert (
            sersic_mp.deflections_yx_2d_from(grid=grid)
            == sersic_lmp.deflections_yx_2d_from(grid=grid)
        ).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllDevVaucouleurs(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
        )
        spherical = ag.lmp.SphDevVaucouleurs(
            centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
        )

        assert elliptical.image_2d_from(grid=grid) == pytest.approx(
            spherical.image_2d_from(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
            spherical.convergence_2d_from(grid=grid), 1.0e-4
        )
        # assert elliptical.potential_2d_from(grid=grid) == spherical.potential_2d_from(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_yx_2d_from(grid=grid),
            spherical.deflections_yx_2d_from(grid=grid),
        )


class TestSersicRadialGradient:
    def test__grid_calculations__same_as_sersic_radial_gradient(self):

        sersic_lp = ag.lmp.EllSersic(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
        )
        sersic_mp = ag.lmp.EllSersicRadialGradient(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=0.5,
        )
        sersic_lmp = ag.lmp.EllSersicRadialGradient(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=0.5,
        )

        assert sersic_lp.image_2d_from(grid=grid) == pytest.approx(
            sersic_lmp.image_2d_from(grid=grid), 1.0e-4
        )
        assert sersic_mp.convergence_2d_from(grid=grid) == pytest.approx(
            sersic_lmp.convergence_2d_from(grid=grid), 1.0e-4
        )
        #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
        assert sersic_mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
            sersic_lmp.deflections_yx_2d_from(grid=grid), 1.0e-4
        )

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllSersicRadialGradient(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
        )
        spherical = ag.lmp.SphSersicRadialGradient(
            centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
        )

        assert elliptical.image_2d_from(grid=grid) == pytest.approx(
            spherical.image_2d_from(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
            spherical.convergence_2d_from(grid=grid), 1.0e-4
        )
        # assert elliptical.potential_2d_from(grid=grid) == spherical.potential_2d_from(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_yx_2d_from(grid=grid),
            spherical.deflections_yx_2d_from(grid=grid),
        )


class TestExponentialRadialGradient:
    def test__grid_calculations__same_as_sersic_radial_gradient(self):

        sersic_lp = ag.lmp.EllExponential(
            elliptical_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6
        )
        sersic_mp = ag.lmp.EllExponentialRadialGradient(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=0.5,
        )
        sersic_lmp = ag.lmp.EllExponentialRadialGradient(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            effective_radius=0.6,
            mass_to_light_ratio=2.0,
            mass_to_light_gradient=0.5,
        )

        assert sersic_lp.image_2d_from(grid=grid) == pytest.approx(
            sersic_lmp.image_2d_from(grid=grid), 1.0e-4
        )
        assert sersic_mp.convergence_2d_from(grid=grid) == pytest.approx(
            sersic_lmp.convergence_2d_from(grid=grid), 1.0e-4
        )
        #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
        assert sersic_mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
            sersic_lmp.deflections_yx_2d_from(grid=grid), 1.0e-4
        )

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllExponentialRadialGradient(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
        )
        spherical = ag.lmp.SphExponentialRadialGradient(
            centre=(0.0, 0.0), intensity=1.0, effective_radius=1.0
        )

        assert elliptical.image_2d_from(grid=grid) == pytest.approx(
            spherical.image_2d_from(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
            spherical.convergence_2d_from(grid=grid), 1.0e-4
        )
        # assert elliptical.potential_2d_from(grid=grid) == spherical.potential_2d_from(grid=grid)
        np.testing.assert_almost_equal(
            elliptical.deflections_yx_2d_from(grid=grid),
            spherical.deflections_yx_2d_from(grid=grid),
        )


class TestSersicCore:
    def test__grid_calculations__same_as_core_sersic(self):

        sersic_lp = ag.lmp.EllSersicCore(
            elliptical_comps=(0.1, 0.05), effective_radius=0.6, sersic_index=2.0
        )
        sersic_mp = ag.lmp.EllSersicCore(
            elliptical_comps=(0.1, 0.05),
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        sersic_lmp = ag.lmp.EllSersicCore(
            elliptical_comps=(0.1, 0.05),
            effective_radius=0.6,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )

        assert (
            sersic_lp.image_2d_from(grid=grid) == sersic_lmp.image_2d_from(grid=grid)
        ).all()
        assert (
            sersic_mp.convergence_2d_from(grid=grid)
            == sersic_lmp.convergence_2d_from(grid=grid)
        ).all()
        #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
        assert (
            sersic_mp.deflections_yx_2d_from(grid=grid)
            == sersic_lmp.deflections_yx_2d_from(grid=grid)
        ).all()

    def test__spherical_and_elliptical_identical(self):

        elliptical = ag.lmp.EllSersicCore(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            effective_radius=1.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )
        spherical = ag.lmp.SphSersicCore(
            centre=(0.0, 0.0),
            effective_radius=1.0,
            sersic_index=2.0,
            mass_to_light_ratio=2.0,
        )

        assert elliptical.image_2d_from(grid=grid) == pytest.approx(
            spherical.image_2d_from(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
            spherical.convergence_2d_from(grid=grid), 1.0e-4
        )
        # assert (elliptical.potential_2d_from(grid=grid) == spherical.potential_2d_from(grid=grid)).all()
        np.testing.assert_almost_equal(
            elliptical.deflections_yx_2d_from(grid=grid),
            spherical.deflections_yx_2d_from(grid=grid),
        )


class TestChameleon:
    def test__grid_calculations__same_as_chameleon(self):

        chameleon_lp = ag.lmp.EllChameleon(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
        )
        chameleon_mp = ag.lmp.EllChameleon(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
            mass_to_light_ratio=2.0,
        )
        chameleon_lmp = ag.lmp.EllChameleon(
            elliptical_comps=(0.1, 0.05),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
            mass_to_light_ratio=2.0,
        )

        assert (
            chameleon_lp.image_2d_from(grid=grid)
            == chameleon_lmp.image_2d_from(grid=grid)
        ).all()
        assert (
            chameleon_mp.convergence_2d_from(grid=grid)
            == chameleon_lmp.convergence_2d_from(grid=grid)
        ).all()
        #    assert (chameleon_mp.potential_2d_from(grid=grid) == chameleon_lmp.potential_2d_from(grid=grid)).all()
        assert (
            chameleon_mp.deflections_yx_2d_from(grid=grid)
            == chameleon_lmp.deflections_yx_2d_from(grid=grid)
        ).all()

    def test__spherical_and_elliptical_identical(self):
        elliptical = ag.lmp.EllChameleon(
            centre=(0.0, 0.0),
            elliptical_comps=(0.0, 0.0),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
            mass_to_light_ratio=2.0,
        )
        spherical = ag.lmp.SphChameleon(
            centre=(0.0, 0.0),
            intensity=1.0,
            core_radius_0=0.1,
            core_radius_1=0.3,
            mass_to_light_ratio=2.0,
        )

        assert elliptical.image_2d_from(grid=grid) == pytest.approx(
            spherical.image_2d_from(grid=grid), 1.0e-4
        )
        assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
            spherical.convergence_2d_from(grid=grid), 1.0e4
        )
        # assert (elliptical.potential_2d_from(grid=grid) == spherical.potential_2d_from(grid=grid)).all()
        np.testing.assert_almost_equal(
            elliptical.deflections_yx_2d_from(grid=grid),
            spherical.deflections_yx_2d_from(grid=grid),
        )
