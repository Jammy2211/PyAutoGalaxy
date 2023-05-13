import numpy as np
import pytest

import autogalaxy as ag


def test__creates_two_smbhs__centres_correct_based_on_angle__init():

    smbh_binary = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=0.0001,
        mass=3.0,
        mass_ratio=2.0,
        redshift_object=0.169,
        redshift_source=0.451
    )

    assert smbh_binary.smbh_0.centre == pytest.approx((8.726646259967217e-07, 0.5), 1e-2)
    assert smbh_binary.smbh_0.mass == pytest.approx(2.0, 1e-2)

    assert smbh_binary.smbh_1.centre == pytest.approx((-8.726646259967217e-07, -0.5), 1e-2)
    assert smbh_binary.smbh_1.mass == pytest.approx(1.0, 1e-2)

    smbh_binary = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=90.0,
        mass=3.0,
        mass_ratio=2.0,
        redshift_object=0.169,
        redshift_source=0.451
    )

    assert smbh_binary.smbh_0.centre == pytest.approx((0.5, 0.0), 1e-2)
    assert smbh_binary.smbh_1.centre == pytest.approx((-0.5, 0.0), 1e-2)

    smbh_binary = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=180.0,
        mass=3.0,
        mass_ratio=2.0,
        redshift_object=0.169,
        redshift_source=0.451
    )

    assert smbh_binary.smbh_0.centre == pytest.approx((0.0, -0.5), 1e-2)
    assert smbh_binary.smbh_1.centre == pytest.approx((0.0, 0.5), 1e-2)

    smbh_binary = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=270.0,
        mass=3.0,
        mass_ratio=2.0,
        redshift_object=0.169,
        redshift_source=0.451
    )

    assert smbh_binary.smbh_0.centre == pytest.approx((-0.5, 0.0), 1e-2)
    assert smbh_binary.smbh_1.centre == pytest.approx((0.5, 0.0), 1e-2)

    smbh_binary = ag.mp.SMBHBinary(
        centre=(0.0, 0.0),
        separation=1.0,
        angle_binary=359.999,
        mass=3.0,
        mass_ratio=2.0,
        redshift_object=0.169,
        redshift_source=0.451
    )

    assert smbh_binary.smbh_0.centre == pytest.approx((-8.726646259517295e-06, 0.5), 1e-2)
    assert smbh_binary.smbh_0.mass == pytest.approx(2.0, 1e-2)

    assert smbh_binary.smbh_1.centre == pytest.approx((8.726646259456063e-06, -0.5), 1e-2)
    assert smbh_binary.smbh_1.mass == pytest.approx(1.0, 1e-2)