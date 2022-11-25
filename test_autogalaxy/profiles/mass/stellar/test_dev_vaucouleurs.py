import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():

    gaussian = ag.mp.DevVaucouleurs()

    deflections = gaussian.deflections_yx_2d_from(grid=np.array([[1.0, 0.0]]))
    deflections_via_cse = gaussian.deflections_2d_via_cse_from(
        grid=np.array([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_cse, 1.0e-4)

    gaussian = ag.mp.DevVaucouleursSph()

    deflections = gaussian.deflections_yx_2d_from(grid=np.array([[1.0, 0.0]]))
    deflections_via_cse = gaussian.deflections_2d_via_cse_from(
        grid=np.array([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_cse, 1.0e-4)


def test__deflections_via_integral_from():
    dev = ag.mp.DevVaucouleurs(
        centre=(0.4, 0.2),
        ell_comps=(0.0180010, 0.0494575),
        intensity=2.0,
        effective_radius=0.8,
        mass_to_light_ratio=3.0,
    )

    deflections = dev.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([(0.1625, 0.1625)])
    )

    assert deflections[0, 0] == pytest.approx(-24.528, 1e-3)
    assert deflections[0, 1] == pytest.approx(-3.37605, 1e-3)


def test__deflections_2d_via_cse_from():
    dev = ag.mp.DevVaucouleurs(
        centre=(0.4, 0.2),
        ell_comps=(0.0180010, 0.0494575),
        intensity=2.0,
        effective_radius=0.8,
        mass_to_light_ratio=3.0,
    )

    deflections_via_integral = dev.deflections_2d_via_integral_from(
        grid=np.array([[0.1625, 0.1625]])
    )
    deflections_via_cse = dev.deflections_2d_via_cse_from(
        grid=np.array([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse, 1.0e-4)

    dev = ag.mp.DevVaucouleurs(
        centre=(0.4, 0.2),
        ell_comps=(0.4180010, 0.694575),
        intensity=2.0,
        effective_radius=0.2,
        mass_to_light_ratio=3.0,
    )

    deflections_via_integral = dev.deflections_2d_via_integral_from(
        grid=np.array([[0.1625, 0.1625]])
    )
    deflections_via_cse = dev.deflections_2d_via_cse_from(
        grid=np.array([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse, 1.0e-4)


def test__convergence_2d_via_mge_from():
    dev = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = dev.convergence_2d_via_mge_from(grid=np.array([[1.0, 0.0]]))

    assert convergence == pytest.approx(5.6697, 1e-3)

    dev = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    convergence = dev.convergence_2d_via_mge_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(7.4455, 1e-3)

    dev = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=4.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    convergence = dev.convergence_2d_via_mge_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 7.4455, 1e-3)

    dev = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=2.0,
    )

    convergence = dev.convergence_2d_via_mge_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 7.4455, 1e-3)

    dev = ag.mp.DevVaucouleurs(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=1.0,
    )

    convergence = dev.convergence_2d_via_mge_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.351797, 1e-3)


def test__convergence_2d_from():
    dev = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = dev.convergence_2d_from(grid=np.array([[1.0, 0.0]]))

    assert convergence == pytest.approx(5.6697, 1e-3)

    dev = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    convergence = dev.convergence_2d_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(7.4455, 1e-3)

    dev = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=4.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    convergence = dev.convergence_2d_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 7.4455, 1e-3)

    dev = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=2.0,
    )

    convergence = dev.convergence_2d_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 7.4455, 1e-3)

    dev = ag.mp.DevVaucouleurs(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=1.0,
    )

    convergence = dev.convergence_2d_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.351797, 1e-3)

    elliptical = ag.mp.DevVaucouleurs(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        mass_to_light_ratio=1.0,
    )

    spherical = ag.mp.DevVaucouleurs(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        mass_to_light_ratio=1.0,
    )

    ell_convergence_2d = elliptical.convergence_2d_from(grid=grid)
    sph_convergence_2d = spherical.convergence_2d_from(grid=grid)

    assert ell_convergence_2d == pytest.approx(sph_convergence_2d, 1.0e-4)
