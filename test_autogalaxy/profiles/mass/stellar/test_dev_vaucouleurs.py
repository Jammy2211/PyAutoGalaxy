import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__dev_vaucouleurs():
    mp = ag.mp.DevVaucouleurs()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_yx_2d_from__dev_vaucouleurs_sph():
    mp = ag.mp.DevVaucouleursSph()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_via_cse_from__config_1():
    mp = ag.mp.DevVaucouleurs(
        centre=(0.4, 0.2),
        ell_comps=(0.0180010, 0.0494575),
        intensity=2.0,
        effective_radius=0.8,
        mass_to_light_ratio=3.0,
    )

    deflections_via_integral = np.array([[-24.528, -3.37605]])
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__deflections_2d_via_cse_from__config_2():
    mp = ag.mp.DevVaucouleurs(
        centre=(0.4, 0.2),
        ell_comps=(0.4180010, 0.694575),
        intensity=2.0,
        effective_radius=0.2,
        mass_to_light_ratio=3.0,
    )

    deflections_via_integral = np.array([[-3.1360329449, 0.1181013046]])
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse.array, 1.0e-4)


def test__convergence_2d_from__dev_vaucouleurs_config_1():
    mp = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(5.6697, 1e-3)


def test__convergence_2d_from__dev_vaucouleurs_config_2():
    mp = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(7.4455, 1e-3)


def test__convergence_2d_from__dev_vaucouleurs_intensity_4():
    mp = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=4.0,
        effective_radius=3.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 7.4455, 1e-3)


def test__convergence_2d_from__dev_vaucouleurs_mass_to_light_2():
    mp = ag.mp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333),
        intensity=2.0,
        effective_radius=3.0,
        mass_to_light_ratio=2.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 7.4455, 1e-3)


def test__convergence_2d_from__dev_vaucouleurs_small_effective_radius():
    mp = ag.mp.DevVaucouleurs(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.351797, 1e-3)


def test__convergence_2d_from__elliptical_vs_spherical():
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
