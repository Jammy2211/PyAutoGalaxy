import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():
    mp = ag.mp.dPIEMassSph(centre=(-0.7, 0.5), b0=5.2, ra=2.0, rs=3.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.033080741, 1e-4)
    assert deflections[0, 1] == pytest.approx(-0.39286169026, 1e-4)

    mp = ag.mp.dPIEMassSph(centre=(-0.1, 0.1), b0=20.0, ra=2.0, rs=3.0)

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(1.4212977207, 1e-4)
    assert deflections[0, 1] == pytest.approx(0.308977765378, 1e-4)

    # First deviation from potential case due to ellipticity

    mp = ag.mp.dPIEMass(
        centre=(0, 0), ell_comps=(0.0, 0.333333), b0=4.0, ra=2.0, rs=3.0
    )

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.21461366, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.10753914, 1e-3)

    elliptical = ag.mp.dPIEMass(
        centre=(1.1, 1.1), ell_comps=(0.0, 0.0), b0=12.0, ra=2.0, rs=3.0
    )
    spherical = ag.mp.dPIEMassSph(centre=(1.1, 1.1), b0=12.0, ra=2.0, rs=3.0)

    assert elliptical.deflections_yx_2d_from(grid=grid).array == pytest.approx(
        spherical.deflections_yx_2d_from(grid=grid).array, 1e-4
    )
