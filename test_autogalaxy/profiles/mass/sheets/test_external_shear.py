import autogalaxy as ag
import numpy as np
import pytest

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__potential_2d_from():
    shear = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    potential = shear.potential_2d_from(grid=np.array([[0.1, 0.1]]))

    assert potential == pytest.approx(np.array([0.001]), 1.0e-4)

    shear = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    potential = shear.potential_2d_from(
        grid=np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    )
    assert potential == pytest.approx(np.array([0.001, 0.004, 0.009]), 1.0e-4)

    shear = ag.mp.ExternalShear(gamma_1=0.1, gamma_2=-0.05)
    potential = shear.potential_2d_from(grid=np.array([[0.1, 0.1]]))

    assert potential == pytest.approx(np.array([-0.0005]), 1.0e-4)


def test__deflections_yx_2d_from():
    shear = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    deflections = shear.deflections_yx_2d_from(grid=np.array([[0.1625, 0.1625]]))
    assert deflections[0, 0] == pytest.approx(0.01625, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.01625, 1e-3)

    shear = ag.mp.ExternalShear(gamma_1=-0.17320, gamma_2=0.1)
    deflections = shear.deflections_yx_2d_from(grid=np.array([[0.1625, 0.1625]]))
    assert deflections[0, 0] == pytest.approx(0.04439, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.011895, 1e-3)

    deflections = shear.deflections_yx_2d_from(
        grid=ag.Grid2D.no_mask(
            values=[
                [[0.1625, 0.1625], [0.1625, 0.1625]],
                [[0.1625, 0.1625], [0.1625, 0.1625]],
            ],
            sub_size=2,
            pixel_scales=(1.0, 1.0),
        )
    )

    assert deflections[0, 0] == pytest.approx(0.04439, 1e-3)
    assert deflections[1, 0] == pytest.approx(0.04439, 1e-3)
    assert deflections[2, 0] == pytest.approx(0.04439, 1e-3)
    assert deflections[3, 0] == pytest.approx(0.04439, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.011895, 1e-3)
    assert deflections[1, 1] == pytest.approx(-0.011895, 1e-3)
    assert deflections[2, 1] == pytest.approx(-0.011895, 1e-3)
    assert deflections[3, 1] == pytest.approx(-0.011895, 1e-3)

    deflections = shear.deflections_yx_2d_from(grid=np.array([[0.1625, 0.1625]]))

    assert deflections[0, 0] == pytest.approx(0.04439, 1e-3)
    assert deflections[0, 1] == pytest.approx(-0.011895, 1e-3)


def test__convergence_returns_zeros():
    shear = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    convergence = shear.convergence_2d_from(grid=np.array([[0.1, 0.1]]))
    assert (convergence == np.array([0.0])).all()

    shear = ag.mp.ExternalShear(gamma_1=0.0, gamma_2=0.1)
    convergence = shear.convergence_2d_from(
        grid=np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
    )
    assert (convergence == np.array([0.0, 0.0, 0.0])).all()

    convergence = shear.convergence_2d_from(
        grid=ag.Grid2D.no_mask(
            values=[[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
            sub_size=2,
            pixel_scales=(1.0, 1.0),
        )
    )

    assert convergence[0] == pytest.approx(0.0, 1e-3)
    assert convergence[1] == pytest.approx(0.0, 1e-3)
    assert convergence[2] == pytest.approx(0.0, 1e-3)
    assert convergence[3] == pytest.approx(0.0, 1e-3)

    convergence = shear.convergence_2d_from(grid=np.array([[1.0, 0.0]]))

    assert convergence[0] == pytest.approx(0.0, 1e-3)
