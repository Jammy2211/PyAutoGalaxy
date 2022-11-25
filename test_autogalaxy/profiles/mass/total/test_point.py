import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():

    # The radial coordinate at (1.0, 1.0) is sqrt(2)
    # This is decomposed into (y,x) angles of sin(45) = cos(45) = sqrt(2) / 2.0
    # Thus, for an EinR of 1.0, the deflection angle is (1.0 / sqrt(2)) * (sqrt(2) / 2.0)

    point_mass = ag.mp.PointMass(centre=(0.0, 0.0), einstein_radius=1.0)

    deflections = point_mass.deflections_yx_2d_from(grid=np.array([[1.0, 1.0]]))

    assert deflections[0, 0] == pytest.approx(0.5, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.5, 1e-3)

    point_mass = ag.mp.PointMass(centre=(0.0, 0.0), einstein_radius=2.0)

    deflections = point_mass.deflections_yx_2d_from(grid=np.array([[1.0, 1.0]]))
    assert deflections[0, 0] == pytest.approx(2.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(2.0, 1e-3)

    point_mass = ag.mp.PointMass(centre=(0.0, 0.0), einstein_radius=1.0)

    deflections = point_mass.deflections_yx_2d_from(grid=np.array([[2.0, 2.0]]))
    assert deflections[0, 0] == pytest.approx(0.25, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.25, 1e-3)

    point_mass = ag.mp.PointMass(centre=(0.0, 0.0), einstein_radius=1.0)

    deflections = point_mass.deflections_yx_2d_from(grid=np.array([[2.0, 1.0]]))
    assert deflections[0, 0] == pytest.approx(0.4, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.2, 1e-3)

    point_mass = ag.mp.PointMass(centre=(0.0, 0.0), einstein_radius=2.0)

    deflections = point_mass.deflections_yx_2d_from(grid=np.array([[4.0, 9.0]]))
    assert deflections[0, 0] == pytest.approx(16.0 / 97.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(36.0 / 97.0, 1e-3)

    point_mass = ag.mp.PointMass(centre=(1.0, 2.0), einstein_radius=1.0)

    deflections = point_mass.deflections_yx_2d_from(grid=np.array([[2.0, 3.0]]))
    assert deflections[0, 0] == pytest.approx(0.5, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.5, 1e-3)
