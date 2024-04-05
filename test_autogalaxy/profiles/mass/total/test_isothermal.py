import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():
    mp = ag.mp.IsothermalSph(centre=(-0.7, 0.5), einstein_radius=1.3)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.21510, 1e-4)
    assert deflections[0, 1] == pytest.approx(-0.46208, 1e-4)

    mp = ag.mp.IsothermalSph(centre=(-0.1, 0.1), einstein_radius=5.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(4.88588, 1e-4)
    assert deflections[0, 1] == pytest.approx(1.06214, 1e-4)

    mp = ag.mp.Isothermal(centre=(0, 0), ell_comps=(0.0, 0.333333), einstein_radius=1.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.79421, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.50734, 1e-3)

    mp = ag.mp.Isothermal(centre=(0, 0), ell_comps=(0.0, 0.333333), einstein_radius=1.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.79421, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.50734, 1e-3)

    elliptical = ag.mp.Isothermal(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), einstein_radius=3.0
    )
    spherical = ag.mp.IsothermalSph(centre=(1.1, 1.1), einstein_radius=3.0)

    assert elliptical.deflections_yx_2d_from(grid=grid) == pytest.approx(
        spherical.deflections_yx_2d_from(grid=grid), 1e-4
    )


def test__convergence_2d_from():
    # eta = 1.0
    # kappa = 0.5 * 1.0 ** 1.0

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.5 * 2.0, 1e-3)

    mp = ag.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), einstein_radius=1.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.5, 1e-3)

    mp = ag.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), einstein_radius=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.5 * 2.0, 1e-3)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.333333), einstein_radius=1.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.66666, 1e-3)

    elliptical = ag.mp.Isothermal(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), einstein_radius=3.0
    )
    spherical = ag.mp.IsothermalSph(centre=(1.1, 1.1), einstein_radius=3.0)

    assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
        spherical.convergence_2d_from(grid=grid), 1e-4
    )


def test__potential_2d_from():
    mp = ag.mp.IsothermalSph(centre=(-0.7, 0.5), einstein_radius=1.3)

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert potential == pytest.approx(1.23435, 1e-3)

    mp = ag.mp.Isothermal(
        centre=(-0.7, 0.5),
        ell_comps=(0.152828, -0.088235),
        einstein_radius=1.3,
    )

    potential = mp.potential_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert potential == pytest.approx(1.19268, 1e-3)

    elliptical = ag.mp.Isothermal(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), einstein_radius=3.0
    )
    spherical = ag.mp.IsothermalSph(centre=(1.1, 1.1), einstein_radius=3.0)

    assert elliptical.potential_2d_from(grid=grid) == pytest.approx(
        spherical.potential_2d_from(grid=grid), 1e-4
    )


def test__shear_yx_2d_from():
    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(0.0, 1e-4)
    assert shear[0, 1] == pytest.approx(-convergence, 1e-4)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[2.0, 1.0]]))
    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[2.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(-(4.0 / 5.0) * convergence, 1e-4)
    assert shear[0, 1] == pytest.approx((3.0 / 5.0) * convergence, 1e-4)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[3.0, 5.0]]))
    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[3.0, 5.0]]))

    assert shear[0, 0] == pytest.approx(-(30.0 / 34.0) * convergence, 1e-4)
    assert shear[0, 1] == pytest.approx(-(16.0 / 34.0) * convergence, 1e-4)

    mp = ag.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), einstein_radius=2.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(0.0, 1e-4)
    assert shear[0, 1] == pytest.approx(-convergence, 1e-4)

    mp = ag.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.3, 0.4), einstein_radius=2.0)

    shear = mp.shear_yx_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert shear[0, 0] == pytest.approx(0.0, 1e-4)
    assert shear[0, 1] == pytest.approx(-1.11803398874, 1e-4)


def test__compare_to_cored_power_law():
    isothermal = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.333333, 0.0), einstein_radius=1.0
    )
    cored_power_law = ag.mp.PowerLawCore(
        centre=(0.0, 0.0),
        ell_comps=(0.333333, 0.0),
        einstein_radius=1.0,
        core_radius=0.0,
    )

    assert isothermal.potential_2d_from(grid=grid) == pytest.approx(
        cored_power_law.potential_2d_from(grid=grid), 1e-3
    )
    assert isothermal.potential_2d_from(grid=grid) == pytest.approx(
        cored_power_law.potential_2d_from(grid=grid), 1e-3
    )
    assert isothermal.deflections_yx_2d_from(grid=grid) == pytest.approx(
        cored_power_law.deflections_yx_2d_from(grid=grid), 1e-3
    )
    assert isothermal.deflections_yx_2d_from(grid=grid) == pytest.approx(
        cored_power_law.deflections_yx_2d_from(grid=grid), 1e-3
    )
