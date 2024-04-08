import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():
    mp = ag.mp.PowerLawBrokenSph(
        centre=(0, 0),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.404076, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.808152, 1e-3)

    deflections = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.5, 1.0], [0.5, 1.0]])
    )

    assert deflections[0, 0] == pytest.approx(0.404076, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.808152, 1e-3)
    assert deflections[1, 0] == pytest.approx(0.404076, 1e-3)
    assert deflections[1, 1] == pytest.approx(0.808152, 1e-3)

    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(0.096225, 0.055555),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.40392, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.811619, 1e-3)

    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(-0.07142, -0.085116),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.4005338, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.8067221, 1e-3)

    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(0.109423, 0.019294),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.399651, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.813372, 1e-3)

    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(-0.216506, -0.125),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.402629, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.798795, 1e-3)


def test__convergence_2d_from():
    mp = ag.mp.PowerLawBrokenSph(
        centre=(0, 0),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert convergence == pytest.approx(0.0355237, 1e-4)

    convergence = mp.convergence_2d_from(
        grid=ag.Grid2DIrregular([[0.5, 1.0], [0.5, 1.0]])
    )

    assert convergence == pytest.approx([0.0355237, 0.0355237], 1e-4)

    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(0.096225, 0.055555),
        einstein_radius=1.0,
        inner_slope=1.5,
        outer_slope=2.5,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert convergence == pytest.approx(0.05006035, 1e-4)

    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(-0.113433, 0.135184),
        einstein_radius=1.0,
        inner_slope=1.8,
        outer_slope=2.2,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert convergence == pytest.approx(0.034768, 1e-4)

    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(0.113433, -0.135184),
        einstein_radius=1.0,
        inner_slope=1.8,
        outer_slope=2.2,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert convergence == pytest.approx(0.03622852, 1e-4)

    mp = ag.mp.PowerLawBroken(
        centre=(0, 0),
        ell_comps=(-0.173789, -0.030643),
        einstein_radius=1.0,
        inner_slope=1.8,
        outer_slope=2.2,
        break_radius=0.1,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    assert convergence == pytest.approx(0.026469, 1e-4)


def test__deflections_yx_2d_from__compare_to_power_law():
    mp = ag.mp.PowerLawBrokenSph(
        centre=(0, 0),
        einstein_radius=2.0,
        inner_slope=1.999,
        outer_slope=2.0001,
        break_radius=0.0001,
    )
    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    # Use of ratio avoids normalization definition difference effects

    broken_yx_ratio = deflections[0, 0] / deflections[0, 1]

    power_law = ag.mp.PowerLawSph(centre=(0, 0), einstein_radius=2.0, slope=2.0)
    deflections = power_law.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.5, 1.0]])
    )

    power_law_yx_ratio = deflections[0, 0] / deflections[0, 1]

    assert broken_yx_ratio == pytest.approx(power_law_yx_ratio, 1.0e-4)

    mp = ag.mp.PowerLawBrokenSph(
        centre=(0, 0),
        einstein_radius=2.0,
        inner_slope=2.399,
        outer_slope=2.4001,
        break_radius=0.0001,
    )
    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.5, 1.0]]))

    # Use of ratio avoids normalization difference effects

    broken_yx_ratio = deflections[0, 0] / deflections[0, 1]

    power_law = ag.mp.PowerLawSph(centre=(0, 0), einstein_radius=2.0, slope=2.4)
    deflections = power_law.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.5, 1.0]])
    )

    power_law_yx_ratio = deflections[0, 0] / deflections[0, 1]

    assert broken_yx_ratio == pytest.approx(power_law_yx_ratio, 1.0e-4)
