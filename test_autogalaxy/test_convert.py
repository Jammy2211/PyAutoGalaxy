import pytest

import autogalaxy as ag

def test__ell_comps_from():

    ell_comps = ag.convert.ell_comps_from(
        axis_ratio=0.00050025012, angle=0.0
    )

    assert ell_comps == pytest.approx((0.0, 0.999), 1.0e-4)

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


def test__shear_gamma_1_2_from():

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(
        magnitude=0.05, angle=0.0
    )

    assert gamma_1 == pytest.approx(0.05, 1.0e-4)
    assert gamma_2 == pytest.approx(0.0, 1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(
        magnitude=0.05, angle=45.0
    )

    assert gamma_1 == pytest.approx(0.0, 1.0e-4)
    assert gamma_2 == pytest.approx(0.05, 1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(
        magnitude=0.05, angle=90.0
    )

    assert gamma_1 == pytest.approx(-0.05, 1.0e-4)
    assert gamma_2 == pytest.approx(0.0, 1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(
        magnitude=0.05, angle=135.0
    )

    assert gamma_1 == pytest.approx(0.0, 1.0e-4)
    assert gamma_2 == pytest.approx(-0.05, 1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(
        magnitude=0.05, angle=180.0
    )

    assert gamma_1 == pytest.approx(0.05, 1.0e-4)
    assert gamma_2 == pytest.approx(0.0, 1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(
        magnitude=0.05, angle=225.0
    )

    assert gamma_1 == pytest.approx(0.0, 1.0e-4)
    assert gamma_2 == pytest.approx(0.05, 1.0e-4)

    gamma_1, gamma_2 = ag.convert.shear_gamma_1_2_from(
        magnitude=0.05, angle=-45.0
    )

    assert gamma_1 == pytest.approx(0.0, 1.0e-4)
    assert gamma_2 == pytest.approx(-0.05, 1.0e-4)

