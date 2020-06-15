import autogalaxy as ag
import numpy as np
import pytest

# Arc second coordinate grid is:.

# [[[-2.0, -2.0], [-2.0, -1.0], [-2.0, 0.0], [-2.0, 1.0], [-2.0, 2.0]],
# [[[-1.0, -2.0], [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0], [-1.0, 2.0]],
# [[[ 0.0, -2.0], [ 0.0, -1.0], [ 0.0, 0.0], [ 0.0, 1.0], [ 0.0, 2.0]],
# [[[ 1.0, -2.0], [ 1.0, -1.0], [ 1.0, 0.0], [ 1.0, 1.0], [ 1.0, 2.0]],
# [[[ 2.0, -2.0], [ 2.0, -1.0], [ 2.0, 0.0], [ 2.0, 1.0], [ 2.0, 2.0]],


def test__centre_light_profile_on_grid_coordinate__peak_flux_is_correct_index():

    grid = ag.Grid.uniform(shape_2d=(5, 5), pixel_scales=1.0, sub_size=1)

    sersic = ag.lp.SphericalSersic(centre=(2.0, -2.0))
    image = sersic.image_from_grid(grid=grid)

    assert image.in_1d.argmax() == 0
    assert np.unravel_index(image.in_2d.argmax(), image.in_2d.shape) == (0, 0)

    sersic = ag.lp.SphericalSersic(centre=(2.0, 2.0))
    image = sersic.image_from_grid(grid=grid)

    assert image.in_1d.argmax() == 4
    assert np.unravel_index(image.in_2d.argmax(), image.in_2d.shape) == (0, 4)

    sersic = ag.lp.SphericalSersic(centre=(-2.0, -2.0))
    image = sersic.image_from_grid(grid=grid)

    assert image.in_1d.argmax() == 20
    assert np.unravel_index(image.in_2d.argmax(), image.in_2d.shape) == (4, 0)

    sersic = ag.lp.SphericalSersic(centre=(-2.0, 2.0))
    image = sersic.image_from_grid(grid=grid)

    assert image.in_1d.argmax() == 24
    assert np.unravel_index(image.in_2d.argmax(), image.in_2d.shape) == (4, 4)


def test__centre_mass_profile_on_grid_coordinate__peak_density_is_correct_index():

    grid = ag.Grid.uniform(shape_2d=(5, 5), pixel_scales=1.0, sub_size=1)

    sis = ag.mp.SphericalIsothermal(centre=(2.0, -2.0))
    density = sis.convergence_from_grid(grid=grid)

    assert density.in_1d.argmax() == 0
    assert np.unravel_index(density.in_2d.argmax(), density.in_2d.shape) == (0, 0)

    sis = ag.mp.SphericalIsothermal(centre=(2.0, 2.0))
    density = sis.convergence_from_grid(grid=grid)

    assert density.in_1d.argmax() == 4
    assert np.unravel_index(density.in_2d.argmax(), density.in_2d.shape) == (0, 4)

    sis = ag.mp.SphericalIsothermal(centre=(-2.0, -2.0))
    density = sis.convergence_from_grid(grid=grid)

    assert density.in_1d.argmax() == 20
    assert np.unravel_index(density.in_2d.argmax(), density.in_2d.shape) == (4, 0)

    sis = ag.mp.SphericalIsothermal(centre=(-2.0, 2.0))
    density = sis.convergence_from_grid(grid=grid)

    assert density.in_1d.argmax() == 24
    assert np.unravel_index(density.in_2d.argmax(), density.in_2d.shape) == (4, 4)


def test__deflection_angles():

    grid = ag.Grid.uniform(shape_2d=(5, 5), pixel_scales=1.0, sub_size=1)

    sis = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    deflections_y_2d = sis.deflections_from_grid(grid=grid).in_2d[:, :, 0]

    assert deflections_y_2d[0, 0] == pytest.approx(-1.0 * deflections_y_2d[4, 0], 1e-2)
    assert deflections_y_2d[1, 1] == pytest.approx(-1.0 * deflections_y_2d[3, 1], 1e-2)
    assert deflections_y_2d[1, 3] == pytest.approx(-1.0 * deflections_y_2d[3, 3], 1e-2)
    assert deflections_y_2d[0, 4] == pytest.approx(-1.0 * deflections_y_2d[4, 4], 1e-2)
    assert deflections_y_2d[2, 0] == pytest.approx(deflections_y_2d[2, 4], 1e-2)

    sis = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)
    deflections_x_2d = sis.deflections_from_grid(grid=grid).in_2d[:, :, 1]

    assert deflections_x_2d[0, 0] == pytest.approx(-1.0 * deflections_x_2d[0, 4], 1e-2)
    assert deflections_x_2d[1, 1] == pytest.approx(-1.0 * deflections_x_2d[1, 3], 1e-2)
    assert deflections_x_2d[3, 1] == pytest.approx(-1.0 * deflections_x_2d[3, 3], 1e-2)
    assert deflections_x_2d[4, 0] == pytest.approx(-1.0 * deflections_x_2d[4, 4], 1e-2)
    assert deflections_x_2d[0, 2] == pytest.approx(deflections_x_2d[4, 2], 1e-2)
