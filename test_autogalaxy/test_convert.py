import pytest

import autogalaxy as ag

def test__axis_ratio_and_angle_from():
    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(
        ell_comps=(0.0, 1.0)
    )

    assert axis_ratio == pytest.approx(0.00050025012, 1.0e-4)
    assert angle == pytest.approx(0.0, 1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(
        ell_comps=(1.0, 0.0)
    )

    assert axis_ratio == pytest.approx(0.00050025012, 1.0e-4)
    assert angle == pytest.approx(45.0, 1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(
        ell_comps=(0.0, -1.0)
    )

    assert axis_ratio == pytest.approx(0.00050025012, 1.0e-4)
    assert angle == pytest.approx(90.0, 1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(
        ell_comps=(-1.0, 0.0)
    )
    
    assert axis_ratio == pytest.approx(0.00050025012, 1.0e-4)
    assert angle == pytest.approx(-45.0, 1.0e-4)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(
        ell_comps=(-1.0, -1.0)
    )

    assert axis_ratio == pytest.approx(0.00050025012, 1.0e-4)
    assert angle == pytest.approx(112.5, 1.0e-4)