import pytest

import autogalaxy as ag

def test__angle_from():
    angle = ag.convert.angle_from(
        ell_comps=(0.0, 1.0)
    )

    assert angle == pytest.approx(0.0, 1.0e-4)

    angle = ag.convert.angle_from(
        ell_comps=(1.0, 0.0)
    )

    assert angle == pytest.approx(45.0, 1.0e-4)

    angle = ag.convert.angle_from(
        ell_comps=(0.0, -1.0)
    )

    assert angle == pytest.approx(90.0, 1.0e-4)

    angle = ag.convert.angle_from(
        ell_comps=(-1.0, 0.0)
    )

    assert angle == pytest.approx(-45.0, 1.0e-4)

    angle = ag.convert.angle_from(
        ell_comps=(-1.0, -1.0)
    )

    assert angle == pytest.approx(112.5, 1.0e-4)