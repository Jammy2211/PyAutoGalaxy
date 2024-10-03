import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_2d_via_integral_from():
    mp = ag.mp.gNFWVirialMassConcSph(
        centre=(0.0, 0.0),
        log10m_vir=12.0,
        c_2=10.0,
        overdens=0.0,
        redshift_object=0.5,
        redshift_source=1.0,
        inner_slope=1.0,
    )

    deflections = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(0.0466231, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.04040671, 1e-3)
