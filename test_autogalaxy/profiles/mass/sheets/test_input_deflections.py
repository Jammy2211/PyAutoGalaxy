import autogalaxy as ag
import numpy as np
import pytest
from autogalaxy import exc


grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from__grid_coordinates_overlap_image_grid_of_deflections():
    deflections_y = ag.Array2D.no_mask(
        values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        pixel_scales=0.1,
        origin=(0.0, 0.0),
    )
    deflections_x = ag.Array2D.no_mask(
        values=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        pixel_scales=0.1,
        origin=(0.0, 0.0),
    )

    image_plane_grid = ag.Grid2D.uniform(
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )

    input_deflections = ag.mp.InputDeflections(
        deflections_y=deflections_y,
        deflections_x=deflections_x,
        image_plane_grid=image_plane_grid,
    )

    grid = ag.Grid2D.uniform(
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )

    deflections = input_deflections.deflections_yx_2d_from(grid=grid)

    assert deflections[:, 0] == pytest.approx(deflections_y, 1.0e-4)
    assert deflections[:, 1] == pytest.approx(deflections_x, 1.0e-4)

    grid = ag.Grid2D.no_mask(
        values=np.array(
            [
                [0.1, 0.0],
                [0.0, 0.0],
                [-0.1, -0.1],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )

    deflections = input_deflections.deflections_yx_2d_from(grid=grid)

    assert deflections[0:3, 0] == pytest.approx([2.0, 5.0, 7.0], 1.0e-4)
    assert deflections[0:3, 1] == pytest.approx([8.0, 5.0, 3.0], 1.0e-4)


def test__deflections_yx_2d_from__grid_coordinates_dont_overlap_image_grid_of_deflections__uses_interpolation():
    deflections_y = ag.Array2D.no_mask(
        values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        pixel_scales=0.1,
        origin=(0.0, 0.0),
    )
    deflections_x = ag.Array2D.no_mask(
        values=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        pixel_scales=0.1,
        origin=(0.0, 0.0),
    )

    image_plane_grid = ag.Grid2D.uniform(
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )

    input_deflections = ag.mp.InputDeflections(
        deflections_y=deflections_y,
        deflections_x=deflections_x,
        image_plane_grid=image_plane_grid,
    )

    grid = ag.Grid2D.no_mask(
        values=np.array(
            [
                [0.05, 0.03],
                [0.02, 0.01],
                [-0.08, -0.04],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )

    deflections = input_deflections.deflections_yx_2d_from(grid=grid)

    assert deflections[0:3, 0] == pytest.approx([3.8, 4.5, 7.0], 1.0e-4)
    assert deflections[0:3, 1] == pytest.approx([6.2, 5.5, 3.0], 1.0e-4)


def test__deflections_yx_2d_from__preload_grid_deflections_used_if_preload_grid_input():
    deflections_y = ag.Array2D.no_mask(
        values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        pixel_scales=0.1,
        origin=(0.0, 0.0),
    )
    deflections_x = ag.Array2D.no_mask(
        values=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        pixel_scales=0.1,
        origin=(0.0, 0.0),
    )

    image_plane_grid = ag.Grid2D.uniform(
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )

    grid = ag.Grid2D.no_mask(
        values=np.array(
            [
                [0.05, 0.03],
                [0.02, 0.01],
                [-0.08, -0.04],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )

    input_deflections = ag.mp.InputDeflections(
        deflections_y=deflections_y,
        deflections_x=deflections_x,
        image_plane_grid=image_plane_grid,
        preload_grid=grid,
    )

    input_deflections.preload_deflections[0, 0] = 1.0

    deflections = input_deflections.deflections_yx_2d_from(grid=grid)

    assert (deflections == input_deflections.preload_deflections).all()


def test__deflections_yx_2d_from__input_grid_extends_beyond_image_plane_grid__raises_exception():
    deflections_y = ag.Array2D.no_mask(
        values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        pixel_scales=0.1,
        origin=(0.0, 0.0),
    )
    deflections_x = ag.Array2D.no_mask(
        values=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        pixel_scales=0.1,
        origin=(0.0, 0.0),
    )

    image_plane_grid = ag.Grid2D.uniform(
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )

    input_deflections = ag.mp.InputDeflections(
        deflections_y=deflections_y,
        deflections_x=deflections_x,
        image_plane_grid=image_plane_grid,
    )

    grid = ag.Grid2D.no_mask(
        values=np.array(
            [
                [0.0999, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )
    input_deflections.deflections_yx_2d_from(grid=grid)

    grid = ag.Grid2D.no_mask(
        values=np.array(
            [
                [0.0, 0.0999],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )
    input_deflections.deflections_yx_2d_from(grid=grid)

    with pytest.raises(exc.ProfileException):
        grid = ag.Grid2D.no_mask(
            values=np.array(
                [
                    [0.11, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
            shape_native=deflections_y.shape_native,
            pixel_scales=deflections_y.pixel_scales,
        )
        input_deflections.deflections_yx_2d_from(grid=grid)

    with pytest.raises(exc.ProfileException):
        grid = ag.Grid2D.no_mask(
            values=np.array(
                [
                    [0.0, 0.11],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
            shape_native=deflections_y.shape_native,
            pixel_scales=deflections_y.pixel_scales,
        )
        input_deflections.deflections_yx_2d_from(grid=grid)


def test__convergence_2d_from_potential_2d_from():
    deflections_y = ag.Array2D.no_mask(
        values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        pixel_scales=0.1,
        origin=(0.0, 0.0),
    )
    deflections_x = ag.Array2D.no_mask(
        values=[[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
        pixel_scales=0.1,
        origin=(0.0, 0.0),
    )

    image_plane_grid = ag.Grid2D.uniform(
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )

    input_deflections = ag.mp.InputDeflections(
        deflections_y=deflections_y,
        deflections_x=deflections_x,
        image_plane_grid=image_plane_grid,
    )

    grid = ag.Grid2D.uniform(
        shape_native=deflections_y.shape_native,
        pixel_scales=deflections_y.pixel_scales,
    )

    convergence = input_deflections.convergence_2d_from(grid=grid)

    convergence_via_jacobian = input_deflections.convergence_2d_via_jacobian_from(
        grid=grid
    )

    assert (convergence == convergence_via_jacobian).all()

    potential = input_deflections.potential_2d_from(grid=grid)

    assert (potential == np.zeros(shape=(9,))).all()
