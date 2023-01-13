import autogalaxy as ag
import numpy as np
import pytest
from autogalaxy import exc


grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():
    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[1.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[2.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(2.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=-1.0)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[1.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(-1.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[2.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(-2.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=2.0)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[2.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(4.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

    # The radial coordinate at (1.0, 1.0) is sqrt(2)
    # This is decomposed into (y,x) angles of sin(45) = cos(45) = sqrt(2) / 2.0
    # Thus, for a mass sheet, the deflection angle is (sqrt(2) * sqrt(2) / 2.0) = 1.0

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[1.0, 1.0]]))
    assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(1.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=2.0)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[1.0, 1.0]]))
    assert deflections[0, 0] == pytest.approx(2.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(2.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=2.0)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[2.0, 2.0]]))
    assert deflections[0, 0] == pytest.approx(4.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(4.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

    # The radial coordinate at (2.0, 1.0) is sqrt(5)
    # This gives an angle of 26.5650512 degrees between the 1.0 and np.sqrt(5) of the triangle
    # This is decomposed into y angle of cos(26.5650512 degrees) = 0.8944271
    # This is decomposed into x angle of sin(26.5650512 degrees) = 0.4472135
    # Thus, for a mass sheet, the deflection angles are:
    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[2.0, 1.0]]))
    assert deflections[0, 0] == pytest.approx(0.8944271 * np.sqrt(5), 1e-3)
    assert deflections[0, 1] == pytest.approx(0.4472135 * np.sqrt(5), 1e-3)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[-1.0, -1.0]]))
    assert deflections[0, 0] == pytest.approx(-1.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(-1.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(1.0, 2.0), kappa=1.0)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[2.0, 3.0]]))
    assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(1.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(1.0, 2.0), kappa=-1.0)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[2.0, 3.0]]))
    assert deflections[0, 0] == pytest.approx(-1.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(-1.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

    deflections = mass_sheet.deflections_yx_2d_from(
        grid=ag.Grid2D.no_mask(
            values=[[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
            sub_size=2,
            pixel_scales=(1.0, 1.0),
        )
    )

    assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
    assert deflections[1, 0] == pytest.approx(1.0, 1e-3)
    assert deflections[2, 0] == pytest.approx(1.0, 1e-3)
    assert deflections[3, 0] == pytest.approx(1.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.0, 1e-3)
    assert deflections[1, 1] == pytest.approx(0.0, 1e-3)
    assert deflections[2, 1] == pytest.approx(0.0, 1e-3)
    assert deflections[3, 1] == pytest.approx(0.0, 1e-3)

    deflections = mass_sheet.deflections_yx_2d_from(grid=np.array([[1.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.0, 1e-3)


def test__convergence_2d_from():

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

    convergence = mass_sheet.convergence_2d_from(grid=np.array([[1.0, 0.0]]))

    assert convergence[0] == pytest.approx(1.0, 1e-3)

    convergence = mass_sheet.convergence_2d_from(
        grid=np.array([[1.0, 0.0], [3.0, 3.0], [5.0, -9.0]])
    )

    assert convergence[0] == pytest.approx(1.0, 1e-3)
    assert convergence[1] == pytest.approx(1.0, 1e-3)
    assert convergence[2] == pytest.approx(1.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=-3.0)

    convergence = mass_sheet.convergence_2d_from(
        grid=np.array([[1.0, 0.0], [3.0, 3.0], [5.0, -9.0]])
    )

    assert convergence[0] == pytest.approx(-3.0, 1e-3)
    assert convergence[1] == pytest.approx(-3.0, 1e-3)
    assert convergence[2] == pytest.approx(-3.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

    convergence = mass_sheet.convergence_2d_from(
        grid=ag.Grid2D.no_mask(
            values=[[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
            sub_size=2,
            pixel_scales=(1.0, 1.0),
        )
    )

    assert convergence[0] == pytest.approx(1.0, 1e-3)
    assert convergence[1] == pytest.approx(1.0, 1e-3)
    assert convergence[2] == pytest.approx(1.0, 1e-3)
    assert convergence[3] == pytest.approx(1.0, 1e-3)

    convergence = mass_sheet.convergence_2d_from(grid=np.array([[1.0, 0.0]]))

    assert convergence[0] == pytest.approx(1.0, 1e-3)


def test__potential_2d_from():
    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

    potential = mass_sheet.potential_2d_from(
        grid=np.array([[1.0, 0.0], [3.0, 3.0], [5.0, -9.0]])
    )

    assert potential[0] == pytest.approx(0.0, 1e-3)
    assert potential[1] == pytest.approx(0.0, 1e-3)
    assert potential[2] == pytest.approx(0.0, 1e-3)

    mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

    potential = mass_sheet.potential_2d_from(
        grid=ag.Grid2D.no_mask(
            values=[[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
            sub_size=2,
            pixel_scales=(1.0, 1.0),
        )
    )

    assert potential[0] == pytest.approx(0.0, 1e-3)
    assert potential[1] == pytest.approx(0.0, 1e-3)
    assert potential[2] == pytest.approx(0.0, 1e-3)
    assert potential[3] == pytest.approx(0.0, 1e-3)

    potential = mass_sheet.potential_2d_from(grid=np.array([[1.0, 0.0]]))

    assert potential[0] == pytest.approx(0.0, 1e-3)
