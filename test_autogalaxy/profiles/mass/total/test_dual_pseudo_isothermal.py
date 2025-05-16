import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():
    mp = ag.mp.dPIESph(centre=(-0.7, 0.5), kappa_scale=1.3, ra=2.0, rs=3.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.033080741, 1e-4)
    assert deflections[0, 1] == pytest.approx(-0.39286169026, 1e-4)

    mp = ag.mp.dPIESph(centre=(-0.1, 0.1), kappa_scale=5.0, ra=2.0, rs=3.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.4212977207, 1e-4)
    assert deflections[0, 1] == pytest.approx(0.308977765378, 1e-4)

    mp = ag.mp.dPIE(
        centre=(0, 0), ell_comps=(0.0, 0.333333), kappa_scale=1.0, ra=2.0, rs=3.0
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.186341843, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.13176363087, 1e-3)

    mp = ag.mp.dPIE(
        centre=(0, 0), ell_comps=(0.0, 0.333333), kappa_scale=1.0, ra=2.0, rs=3.0
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.186341843, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.13176363087, 1e-3)

    elliptical = ag.mp.dPIE(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), kappa_scale=3.0, ra=2.0, rs=3.0
    )
    spherical = ag.mp.dPIESph(centre=(1.1, 1.1), kappa_scale=3.0, ra=2.0, rs=3.0)

    assert elliptical.deflections_yx_2d_from(grid=grid) == pytest.approx(
        spherical.deflections_yx_2d_from(grid=grid), 1e-4
    )


def test__convergence_2d_from():
    # eta = 1.0
    # kappa = 0.5 * 1.0 ** 1.0

    mp = ag.mp.dPIESph(centre=(0.0, 0.0), kappa_scale=2.0, ra=2.0, rs=3.0)

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(1.57182995, 1e-3)

    mp = ag.mp.dPIE(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), kappa_scale=1.0, ra=2.0, rs=3.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.78591498, 1e-3)

    mp = ag.mp.dPIE(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), kappa_scale=2.0, ra=2.0, rs=3.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(1.57182995, 1e-3)

    mp = ag.mp.dPIE(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.333333), kappa_scale=1.0, ra=2.0, rs=3.0
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.87182837, 1e-3)

    elliptical = ag.mp.dPIE(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), kappa_scale=3.0, ra=2.0, rs=3.0
    )
    spherical = ag.mp.dPIESph(centre=(1.1, 1.1), kappa_scale=3.0, ra=2.0, rs=3.0)

    assert elliptical.convergence_2d_from(grid=grid) == pytest.approx(
        spherical.convergence_2d_from(grid=grid), 1e-4
    )
