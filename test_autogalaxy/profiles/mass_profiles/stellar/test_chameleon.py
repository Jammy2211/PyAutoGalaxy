import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_2d_via_analytic_from():
    chameleon = ag.mp.EllChameleon(
        centre=(-0.4, -0.2),
        elliptical_comps=(-0.07142, -0.085116),
        intensity=5.0,
        core_radius_0=0.2,
        core_radius_1=0.4,
        mass_to_light_ratio=3.0,
    )

    deflections = chameleon.deflections_2d_via_analytic_from(
        grid=np.array([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(2.12608, 1e-3)
    assert deflections[0, 1] == pytest.approx(1.55252, 1e-3)


def test__deflections_yx_2d_from():

    sersic_core = ag.mp.EllChameleon()

    deflections = sersic_core.deflections_yx_2d_from(grid=np.array([[1.0, 0.0]]))
    deflections_via_integral = sersic_core.deflections_2d_via_analytic_from(
        grid=np.array([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral, 1.0e-4)

    sersic_core = ag.mp.SphChameleon()

    deflections = sersic_core.deflections_yx_2d_from(grid=np.array([[1.0, 0.0]]))
    deflections_via_integral = sersic_core.deflections_2d_via_analytic_from(
        grid=np.array([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral, 1.0e-4)


def test__spherical_and_elliptical_identical():

    elliptical = ag.mp.EllChameleon(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.0),
        intensity=1.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.SphChameleon(
        centre=(0.0, 0.0), intensity=1.0, mass_to_light_ratio=1.0
    )

    np.testing.assert_almost_equal(
        elliptical.deflections_yx_2d_from(grid=grid),
        spherical.deflections_yx_2d_from(grid=grid),
    )


def test__convergence_2d_from():

    chameleon = ag.mp.EllChameleon(
        elliptical_comps=(0.0, 0.0),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
        mass_to_light_ratio=2.0,
    )

    convergence = chameleon.convergence_2d_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 0.018605, 1e-3)

    chameleon = ag.mp.EllChameleon(
        elliptical_comps=(0.5, 0.0),
        intensity=3.0,
        core_radius_0=0.2,
        core_radius_1=0.4,
        mass_to_light_ratio=1.0,
    )

    convergence = chameleon.convergence_2d_from(grid=np.array([[0.0, 1.5]]))

    assert convergence == pytest.approx(0.007814, 1e-3)

    elliptical = ag.mp.EllChameleon(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.0),
        intensity=1.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.SphChameleon(
        centre=(0.0, 0.0), intensity=1.0, mass_to_light_ratio=1.0
    )

    assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
        spherical.convergence_2d_from(grid=grid), 1.0e-4
    )
