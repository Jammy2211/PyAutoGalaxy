import numpy as np
import pytest

import autoarray as aa
import automodel as am
from automodel.inversion import mappers
from test_autoarray.mock.mock_grids import MockPixelizationGrid
from test_automodel.mock.mock_inversion import MockGeometry


def grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers):
    def compute_squared_separation(coordinate1, coordinate2):
        """Computes the squared separation of two grid (no square root for efficiency)"""
        return (coordinate1[0] - coordinate2[0]) ** 2 + (
            coordinate1[1] - coordinate2[1]
        ) ** 2

    image_pixels = grid.shape[0]

    image_to_pixelization = np.zeros((image_pixels,))

    for image_index, image_coordinate in enumerate(grid):
        distances = list(
            map(
                lambda centers: compute_squared_separation(image_coordinate, centers),
                pixel_centers,
            )
        )

        image_to_pixelization[image_index] = np.argmin(distances)

    return image_to_pixelization


@pytest.fixture(name="three_pixels")
def make_three_pixels():
    return np.array([[0, 0], [0, 1], [1, 0]])


@pytest.fixture(name="five_pixels")
def make_five_pixels():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])


class TestRectangularMapper:
    class TestImageAndSubToPixelization:
        def test__3x3_grid_of_pixel_grid__1_coordinate_per_square_pixel__in_centre_of_pixels(
            self
        ):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

            pixelization_grid = np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=9,
                shape=(3, 3),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
            ]

        def test__3x3_grid_of_pixel_grid__1_coordinate_per_square_pixel__near_edges_of_pixels(
            self
        ):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

            pixelization_grid = np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [-0.32, -1.0],
                    [-0.32, 0.32],
                    [0.0, 1.0],
                    [-0.34, -0.34],
                    [-0.34, 0.325],
                    [-1.0, 1.0],
                ]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=9,
                shape=(3, 3),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
            ]

        def test__3x3_grid_of_pixel_grid__add_multiple_grid_to_1_pixel_pixel(self):
            #                  _ _ _
            # -1.0 to -(1/3)  |_|_|_|
            # -(1/3) to (1/3) |_|_|_|
            #  (1/3) to 1.0   |_|_|_|

            pixelization_grid = np.array(
                [
                    [1.0, -1.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [-1.0, -1.0],
                    [0.0, 0.0],
                    [-1.0, 1.0],
                ]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=9,
                shape=(3, 3),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 4, 2, 4, 4, 4, 6, 4, 8])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [],
                [2],
                [],
                [1, 3, 4, 5, 7],
                [],
                [6],
                [],
                [8],
            ]

        def test__4x3_grid_of_pixel_grid__1_coordinate_in_each_pixel(self):
            #   _ _ _
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|

            # Boundaries for column pixel 0 -1.0 to -(1/3)
            # Boundaries for column pixel 1 -(1/3) to (1/3)
            # Boundaries for column pixel 2  (1/3) to 1.0

            # Bounadries for row pixel 0 -1.0 to -0.5
            # Bounadries for row pixel 1 -0.5 to 0.0
            # Bounadries for row pixel 2  0.0 to 0.5
            # Bounadries for row pixel 3  0.5 to 1.0

            pixelization_grid = np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.5, -1.0],
                    [-0.5, 1.0],
                    [-1.0, 1.0],
                ]
            )

            pix = am.pix.Rectangular(shape_2d=(4, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=12,
                shape=(4, 3),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 2, 3, 8, 11])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [2],
                [3],
                [],
                [],
                [],
                [],
                [4],
                [],
                [],
                [5],
            ]

        def test__3x4_grid_of_pixel_grid__1_coordinate_in_each_pixel(self):
            #   _ _ _ _
            #  |_|_|_|_|
            #  |_|_|_|_|
            #  |_|_|_|_|

            # Boundaries for row pixel 0 -1.0 to -(1/3)
            # Boundaries for row pixel 1 -(1/3) to (1/3)
            # Boundaries for row pixel 2  (1/3) to 1.0

            # Bounadries for column pixel 0 -1.0 to -0.5
            # Bounadries for column pixel 1 -0.5 to 0.0
            # Bounadries for column pixel 2  0.0 to 0.5
            # Bounadries for column pixel 3  0.5 to 1.0

            pixelization_grid = np.array(
                [
                    [1.0, -1.0],
                    [1.0, -0.49],
                    [1.0, 0.01],
                    [0.32, 0.01],
                    [-0.34, -0.01],
                    [-1.0, 1.0],
                ]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 4))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=12,
                shape=(3, 4),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 2, 6, 9, 11])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [2],
                [],
                [],
                [],
                [3],
                [],
                [],
                [4],
                [],
                [5],
            ]

        def test__3x3_grid__change_arcsecond_dimensions_size__grid_adapts_accordingly(
            self
        ):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.5 to -0.5
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 -0.5 to 0.5
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2  0.5 to 1.5

            pixelization_grid = np.array(
                [[1.5, -1.5], [1.0, 0.0], [1.0, 0.6], [-1.4, 0.0], [-1.5, 1.5]]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=9,
                shape=(3, 3),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 2, 7, 8])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [2],
                [],
                [],
                [],
                [],
                [3],
                [4],
            ]

        def test__3x3_grid__change_arcsecond_dimensions__not_symmetric(self):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.5 to -0.5
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 -0.5 to 0.5
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2  0.5 to 1.5

            pixelization_grid = np.array(
                [[1.0, -1.5], [1.0, -0.49], [0.32, -1.5], [0.32, 0.51], [-1.0, 1.5]]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=9,
                shape=(3, 3),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 3, 5, 8])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [],
                [2],
                [],
                [3],
                [],
                [],
                [4],
            ]

        def test__4x3_grid__change_arcsecond_dimensions__not_symmetric(self):
            #   _ _ _
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|

            pixelization_grid = np.array(
                [[1.0, -1.5], [1.0, -0.49], [0.49, -1.5], [-0.6, 0.0], [-1.0, 1.5]]
            )

            pix = am.pix.Rectangular(shape_2d=(4, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=12,
                shape=(4, 3),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 3, 10, 11])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [],
                [2],
                [],
                [],
                [],
                [],
                [],
                [],
                [3],
                [4],
            ]

        def test__3x4_grid__change_arcsecond_dimensions__not_symmetric(self):
            #   _ _ _ _
            #  |_|_|_|_|
            #  |_|_|_|_|
            #  |_|_|_|_|

            pixelization_grid = np.array(
                [[1.0, -1.5], [1.0, -0.49], [0.32, -1.5], [-0.34, 0.49], [-1.0, 1.5]]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 4))

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            pix = mappers.RectangularMapper(
                pixels=12,
                shape=(3, 4),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 4, 10, 11])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [],
                [],
                [2],
                [],
                [],
                [],
                [],
                [],
                [3],
                [4],
            ]

        def test__different_image_and_sub_grids(self):
            #                  _ _ _
            # -1.0 to -(1/3)  |_|_|_|
            # -(1/3) to (1/3) |_|_|_|
            #  (1/3) to 1.0   |_|_|_|

            pixelization_grid = np.array(
                [
                    [1.0, -1.0],
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [-1.0, -1.0],
                    [0.0, 0.0],
                    [-1.0, 1.0],
                ]
            )

            pixelization_sub_grid = np.array(
                [
                    [1.0, -1.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, 0.0],
                    [-1.0, 1.0],
                ]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_sub_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=9,
                shape=(3, 3),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
            ]

        def test__3x3_grid_of_pixel_grid___shift_coordinates_to_new_centre__centre_adjusts_based_on_grid(
            self
        ):
            #   _ _ _
            #  |_|_|_| Boundaries for pixels x = 0 and y = 0  -1.0 to -(1/3)
            #  |_|_|_| Boundaries for pixels x = 1 and y = 1 - (1/3) to (1/3)
            #  |_|_|_| Boundaries for pixels x = 2 and y = 2 - (1/3)" to 1.0"

            pixelization_grid = np.array(
                [
                    [2.0, 0.0],
                    [2.0, 1.0],
                    [2.0, 2.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 2.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 2.0],
                ]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=9,
                shape=(3, 3),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
            ).all()

            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
            ]

        def test__4x3_grid__non_symmetric_centre_shift(self):
            #   _ _ _
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|
            #  |_|_|_|

            pixelization_grid = np.array(
                [[3.0, -0.5], [3.0, 0.51], [2.49, -0.5], [1.4, 1.0], [1.0, 2.5]]
            )

            pix = am.pix.Rectangular(shape_2d=(4, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            grid = MockPixelizationGrid(
                pixelization_grid,
                mask_1d_index_for_sub_mask_1d_index=np.ones((1)),
                sub_size=1,
            )

            pix = mappers.RectangularMapper(
                pixels=12,
                shape=(4, 3),
                grid=grid,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            assert (
                pix.pixelization_1d_index_for_sub_mask_1d_index
                == np.array([0, 1, 3, 10, 11])
            ).all()
            assert pix.all_sub_mask_1d_indexes_for_pixelization_1d_index == [
                [0],
                [1],
                [],
                [2],
                [],
                [],
                [],
                [],
                [],
                [],
                [3],
                [4],
            ]

    class TestReconstructedPixelization:
        def test__3x3_pixelization__solution_vector_ascending(self):
            pixelization_grid = np.array(
                [
                    [2.0, -1.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-2.0, -1.0],
                    [-2.0, 0.0],
                    [-2.0, 1.0],
                ]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            pix = mappers.RectangularMapper(
                pixels=9,
                shape=(3, 3),
                grid=None,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
            )

            recon_pix = pix.reconstructed_pixelization_from_solution_vector(
                solution_vector=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            )

            assert (
                recon_pix.in_2d
                == np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
            ).all()
            assert recon_pix.pixel_scales == pytest.approx((4.0 / 3.0, 2.0 / 3.0), 1e-2)
            assert recon_pix.origin == (0.0, 0.0)

        def test__compare_to_imaging_util(self):
            pix = mappers.RectangularMapper(
                pixels=9,
                shape=(4, 3),
                grid=None,
                pixelization_grid=None,
                geometry=MockGeometry(),
            )

            solution = np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0]
            )
            recon_pix = pix.reconstructed_pixelization_from_solution_vector(
                solution_vector=solution
            )
            recon_pix_util = aa.util.array.sub_array_2d_from_sub_array_1d(
                sub_array_1d=solution,
                mask=np.full(fill_value=False, shape=(4, 3)),
                sub_size=1,
            )
            assert (recon_pix.in_2d == recon_pix_util).all()
            assert recon_pix.shape_2d == (4, 3)

            pix = mappers.RectangularMapper(
                pixels=9,
                shape=(3, 4),
                grid=None,
                pixelization_grid=None,
                geometry=MockGeometry(),
            )
            solution = np.array(
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0]
            )
            recon_pix = pix.reconstructed_pixelization_from_solution_vector(
                solution_vector=solution
            )
            recon_pix_util = aa.util.array.sub_array_2d_from_sub_array_1d(
                sub_array_1d=solution,
                mask=np.full(fill_value=False, shape=(3, 4)),
                sub_size=1,
            )
            assert (recon_pix.in_2d == recon_pix_util).all()
            assert recon_pix.shape_2d == (3, 4)

    class TestPixelScales:

        def test__pixel_signals__compare_to_mapper_util(self, grid_7x7, image_7x7):

            pixelization_grid = np.array(
                [
                    [2.0, -1.0],
                    [2.0, 0.0],
                    [2.0, 1.0],
                    [0.0, -1.0],
                    [0.0, 0.0],
                    [0.0, 1.0],
                    [-2.0, -1.0],
                    [-2.0, 0.0],
                    [-2.0, 1.0],
                ]
            )

            pix = am.pix.Rectangular(shape_2d=(3, 3))

            geometry = pix.geometry_from_grid(grid=pixelization_grid)

            mapper = mappers.RectangularMapper(
                pixels=6,
                shape=(3, 3),
                grid=grid_7x7,
                pixelization_grid=pixelization_grid,
                geometry=geometry,
                hyper_image=image_7x7
            )

            pixel_signals = mapper.pixel_signals_from_signal_scale(
                signal_scale=2.0
            )

            pixel_signals_util = am.util.mapper.adaptive_pixel_signals_from_images(
                pixels=6,
                signal_scale=2.0,
                pixelization_1d_index_for_sub_mask_1d_index=mapper.pixelization_1d_index_for_sub_mask_1d_index,
                mask_1d_index_for_sub_mask_1d_index=grid_7x7.regions._mask_1d_index_for_sub_mask_1d_index,
                hyper_image=image_7x7,
            )

            assert (pixel_signals == pixel_signals_util).all()


class TestVoronoiMapper:
    class TestSubToPixelizationViaNearestNeighborsForTesting:
        def test__grid_to_pixel_pixels_via_nearest_neighbour__case1__correct_pairs(
            self
        ):
            pixel_centers = np.array(
                [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
            )
            grid = np.array([[1.1, 1.1], [-1.1, 1.1], [-1.1, -1.1], [1.1, -1.1]])

            sub_to_pix = grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers)

            assert sub_to_pix[0] == 0
            assert sub_to_pix[1] == 1
            assert sub_to_pix[2] == 2
            assert sub_to_pix[3] == 3

        def test__grid_to_pixel_pixels_via_nearest_neighbour___case2__correct_pairs(
            self
        ):
            pixel_centers = np.array(
                [[1.0, 1.0], [-1.0, 1.0], [-1.0, -1.0], [1.0, -1.0]]
            )
            grid = np.array(
                [
                    [1.1, 1.1],
                    [-1.1, 1.1],
                    [-1.1, -1.1],
                    [1.1, -1.1],
                    [0.9, -0.9],
                    [-0.9, -0.9],
                    [-0.9, 0.9],
                    [0.9, 0.9],
                ]
            )

            sub_to_pix = grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers)

            assert sub_to_pix[0] == 0
            assert sub_to_pix[1] == 1
            assert sub_to_pix[2] == 2
            assert sub_to_pix[3] == 3
            assert sub_to_pix[4] == 3
            assert sub_to_pix[5] == 2
            assert sub_to_pix[6] == 1
            assert sub_to_pix[7] == 0

        def test__grid_to_pixel_pixels_via_nearest_neighbour___case3__correct_pairs(
            self
        ):
            pixel_centers = np.array(
                [
                    [1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, -1.0],
                    [1.0, -1.0],
                    [0.0, 0.0],
                    [2.0, 2.0],
                ]
            )
            grid = np.array(
                [
                    [0.1, 0.1],
                    [-0.1, -0.1],
                    [0.49, 0.49],
                    [0.51, 0.51],
                    [1.01, 1.01],
                    [1.51, 1.51],
                ]
            )

            sub_to_pix = grid_to_pixel_pixels_via_nearest_neighbour(grid, pixel_centers)

            assert sub_to_pix[0] == 4
            assert sub_to_pix[1] == 4
            assert sub_to_pix[2] == 4
            assert sub_to_pix[3] == 0
            assert sub_to_pix[4] == 0
            assert sub_to_pix[5] == 5

    class TestSubToPixelization:
        def test__sub_to_pix_of_mapper_matches_nearest_neighbor_calculation(self, grid_7x7):
            pixel_centers = np.array(
                [[0.1, 0.1], [1.1, 0.1], [2.1, 0.1], [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]]
            )

            sub_to_pix_nearest_neighbour = grid_to_pixel_pixels_via_nearest_neighbour(
                grid_7x7, pixel_centers
            )

            nearest_pixelization_1d_index_for_mask_1d_index = np.array(
                [0, 0, 1, 0, 0, 1, 2, 2, 3]
            )

            pixelization_grid = MockPixelizationGrid(
                arr=pixel_centers,
                nearest_pixelization_1d_index_for_mask_1d_index=nearest_pixelization_1d_index_for_mask_1d_index,
            )

            pix = am.pix.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(pixel_centers)
            pixel_neighbors, pixel_neighbors_size = pix.neighbors_from_pixels_and_ridge_points(
                pixels=6, ridge_points=voronoi.ridge_points
            )

            mapper = mappers.VoronoiMapper(
                pixels=6,
                grid=grid_7x7,
                pixelization_grid=pixelization_grid,
                voronoi=voronoi,
                geometry=MockGeometry(
                    pixel_centres=pixel_centers,
                    pixel_neighbors=pixel_neighbors,
                    pixel_neighbors_size=pixel_neighbors_size,
                ),
            )

            assert (
                mapper.pixelization_1d_index_for_sub_mask_1d_index
                == sub_to_pix_nearest_neighbour
            ).all()

    class TestPixelScales:

        def test__pixel_scales_work_for_voronoi_mapper(self, grid_7x7, image_7x7):
            pixel_centers = np.array(
                [[0.1, 0.1], [1.1, 0.1], [2.1, 0.1], [0.1, 1.1], [1.1, 1.1], [2.1, 1.1]]
            )

            nearest_pixelization_1d_index_for_mask_1d_index = np.array(
                [0, 0, 1, 0, 0, 1, 2, 2, 3]
            )

            pixelization_grid = MockPixelizationGrid(
                arr=pixel_centers,
                nearest_pixelization_1d_index_for_mask_1d_index=nearest_pixelization_1d_index_for_mask_1d_index,
            )

            pix = am.pix.Voronoi()
            voronoi = pix.voronoi_from_pixel_centers(pixel_centers)
            pixel_neighbors, pixel_neighbors_size = pix.neighbors_from_pixels_and_ridge_points(
                pixels=6, ridge_points=voronoi.ridge_points
            )

            mapper = mappers.VoronoiMapper(
                pixels=6,
                grid=grid_7x7,
                pixelization_grid=pixelization_grid,
                voronoi=voronoi,
                geometry=MockGeometry(
                    pixel_centres=pixel_centers,
                    pixel_neighbors=pixel_neighbors,
                    pixel_neighbors_size=pixel_neighbors_size,
                ),
                hyper_image=image_7x7
            )

            pixel_signals = mapper.pixel_signals_from_signal_scale(
                signal_scale=2.0
            )

            pixel_signals_util = am.util.mapper.adaptive_pixel_signals_from_images(
                pixels=6,
                signal_scale=2.0,
                pixelization_1d_index_for_sub_mask_1d_index=mapper.pixelization_1d_index_for_sub_mask_1d_index,
                mask_1d_index_for_sub_mask_1d_index=grid_7x7.regions._mask_1d_index_for_sub_mask_1d_index,
                hyper_image=image_7x7,
            )

            assert (pixel_signals == pixel_signals_util).all()