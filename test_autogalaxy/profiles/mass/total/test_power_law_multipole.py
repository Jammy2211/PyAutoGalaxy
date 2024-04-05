import autogalaxy as ag
import numpy as np
import pytest

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():
    multipole = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.1, 0.2),
        einstein_radius=2.0,
        slope=2.2,
        multipole_comps=(0.1, 0.2),
    )

    deflections = multipole.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections[0, 0] == pytest.approx(-0.072229375535, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.2089041286, 1e-3)

    multipole = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.2, 0.3),
        einstein_radius=3.0,
        slope=1.7,
        multipole_comps=(0.2, 0.3),
    )

    deflections = multipole.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections[0, 0] == pytest.approx(-0.2532106, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.5514646, 1e-3)


def test__convergence_2d_from():
    multipole = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.1, 0.2),
        einstein_radius=2.0,
        slope=2.2,
        multipole_comps=(0.1, 0.2),
    )

    convergence = multipole.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence[0] == pytest.approx(0.25958037, 1e-3)

    multipole = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.2, 0.3),
        einstein_radius=3.0,
        slope=1.7,
        multipole_comps=(0.2, 0.3),
    )

    convergence = multipole.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence[0] == pytest.approx(0.2875647, 1e-3)


def test__potential_2d_from():
    multipole = ag.mp.PowerLawMultipole(
        m=4,
        centre=(0.1, 0.2),
        einstein_radius=2.0,
        slope=2.2,
        multipole_comps=(0.1, 0.2),
    )

    potential = multipole.potential_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert potential[0] == pytest.approx(0.0, 1e-3)
