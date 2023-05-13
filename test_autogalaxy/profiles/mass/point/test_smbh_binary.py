import numpy as np
import pytest

import autogalaxy as ag


def test__x2_smbhs__centres_correct_based_on_angle__init():
    smbh_binary = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=0.0001,
    )

    assert smbh_binary.smbh_0.centre == pytest.approx(
        (8.726646259967217e-07, 0.5), 1e-2
    )
    assert smbh_binary.smbh_1.centre == pytest.approx(
        (-8.726646259967217e-07, -0.5), 1e-2
    )

    smbh_binary = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=90.0,
        mass=3.0,
        mass_ratio=2.0,
        redshift_object=0.169,
        redshift_source=0.451,
    )

    assert smbh_binary.smbh_0.centre == pytest.approx((0.5, 0.0), 1e-2)
    assert smbh_binary.smbh_1.centre == pytest.approx((-0.5, 0.0), 1e-2)

    smbh_binary = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=180.0,
    )

    assert smbh_binary.smbh_0.centre == pytest.approx((0.0, -0.5), 1e-2)
    assert smbh_binary.smbh_1.centre == pytest.approx((0.0, 0.5), 1e-2)

    smbh_binary = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=270.0,
    )

    assert smbh_binary.smbh_0.centre == pytest.approx((-0.5, 0.0), 1e-2)
    assert smbh_binary.smbh_1.centre == pytest.approx((0.5, 0.0), 1e-2)

    smbh_binary = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=359.999,
    )

    assert smbh_binary.smbh_0.centre == pytest.approx(
        (-8.726646259517295e-06, 0.5), 1e-2
    )
    assert smbh_binary.smbh_1.centre == pytest.approx(
        (8.726646259456063e-06, -0.5), 1e-2
    )


def test__x2_smbhs__centres_shifted_based_on_centre__init():
    smbh_binary = ag.mp.SMBHBinary(
        centre=(3.0, 1.0),
        separation=1.0,
        angle_binary=0.0001,
    )

    assert smbh_binary.smbh_0.centre == pytest.approx(
        (3.0, 1.5), 1e-2
    )
    assert smbh_binary.smbh_1.centre == pytest.approx(
        (3.0, 0.5), 1e-2
    )

    smbh_binary = ag.mp.SMBHBinary(
        centre=(-3.0, -1.0),
        separation=1.0,
        angle_binary=0.0001,
    )

    assert smbh_binary.smbh_0.centre == pytest.approx(
        (-3.0, -0.5), 1e-2
    )
    assert smbh_binary.smbh_1.centre == pytest.approx(
        (-3.0, -1.5), 1e-2
    )


def test__x2_smbhs__masses_corrected_based_on_mass_and_ratio__init():
    smbh_binary = ag.mp.SMBHBinary(
        mass=3.0,
        mass_ratio=2.0,
    )

    assert smbh_binary.smbh_0.mass == pytest.approx(2.0, 1e-2)
    assert smbh_binary.smbh_1.mass == pytest.approx(1.0, 1e-2)

    smbh_binary = ag.mp.SMBHBinary(
        mass=3.0,
        mass_ratio=0.5,
    )

    assert smbh_binary.smbh_0.mass == pytest.approx(1.0, 1e-2)
    assert smbh_binary.smbh_1.mass == pytest.approx(2.0, 1e-2)
