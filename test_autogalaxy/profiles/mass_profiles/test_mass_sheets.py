import autogalaxy as ag
import numpy as np
import pytest
from autogalaxy import exc


grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


class TestMassSheet:
    def test__convergence__correct_values(self):

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        convergence = mass_sheet.convergence_from_grid(grid=np.array([[1.0, 0.0]]))

        assert convergence[0] == pytest.approx(1.0, 1e-3)

        convergence = mass_sheet.convergence_from_grid(
            grid=np.array([[1.0, 0.0], [3.0, 3.0], [5.0, -9.0]])
        )

        assert convergence[0] == pytest.approx(1.0, 1e-3)
        assert convergence[1] == pytest.approx(1.0, 1e-3)
        assert convergence[2] == pytest.approx(1.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=-3.0)

        convergence = mass_sheet.convergence_from_grid(
            grid=np.array([[1.0, 0.0], [3.0, 3.0], [5.0, -9.0]])
        )

        assert convergence[0] == pytest.approx(-3.0, 1e-3)
        assert convergence[1] == pytest.approx(-3.0, 1e-3)
        assert convergence[2] == pytest.approx(-3.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        convergence = mass_sheet.convergence_from_grid(
            grid=ag.Grid2D.manual_native(
                [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
                sub_size=2,
                pixel_scales=(1.0, 1.0),
            )
        )

        assert convergence[0] == pytest.approx(1.0, 1e-3)
        assert convergence[1] == pytest.approx(1.0, 1e-3)
        assert convergence[2] == pytest.approx(1.0, 1e-3)
        assert convergence[3] == pytest.approx(1.0, 1e-3)

        convergence = mass_sheet.convergence_from_grid(grid=np.array([[1.0, 0.0]]))

        assert convergence[0] == pytest.approx(1.0, 1e-3)

    def test__potential__correct_values(self):
        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        potential = mass_sheet.potential_from_grid(
            grid=np.array([[1.0, 0.0], [3.0, 3.0], [5.0, -9.0]])
        )

        assert potential[0] == pytest.approx(0.0, 1e-3)
        assert potential[1] == pytest.approx(0.0, 1e-3)
        assert potential[2] == pytest.approx(0.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        potential = mass_sheet.potential_from_grid(
            grid=ag.Grid2D.manual_native(
                [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
                sub_size=2,
                pixel_scales=(1.0, 1.0),
            )
        )

        assert potential[0] == pytest.approx(0.0, 1e-3)
        assert potential[1] == pytest.approx(0.0, 1e-3)
        assert potential[2] == pytest.approx(0.0, 1e-3)
        assert potential[3] == pytest.approx(0.0, 1e-3)

        potential = mass_sheet.potential_from_grid(grid=np.array([[1.0, 0.0]]))

        assert potential[0] == pytest.approx(0.0, 1e-3)

    def test__deflections__correct_values(self):
        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(2.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=-1.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(-1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(-2.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=2.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(4.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        # The radial coordinate at (1.0, 1.0) is sqrt(2)
        # This is decomposed into (y,x) angles of sin(45) = cos(45) = sqrt(2) / 2.0
        # Thus, for a mass sheet, the deflection angle is (sqrt(2) * sqrt(2) / 2.0) = 1.0

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=2.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        assert deflections[0, 0] == pytest.approx(2.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(2.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=2.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 2.0]]))
        assert deflections[0, 0] == pytest.approx(4.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(4.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        # The radial coordinate at (2.0, 1.0) is sqrt(5)
        # This gives an angle of 26.5650512 degrees between the 1.0 and np.sqrt(5) of the triangle
        # This is decomposed into y angle of cos(26.5650512 degrees) = 0.8944271
        # This is decomposed into x angle of sin(26.5650512 degrees) = 0.4472135
        # Thus, for a mass sheet, the deflection angles are:
        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 1.0]]))
        assert deflections[0, 0] == pytest.approx(0.8944271 * np.sqrt(5), 1e-3)
        assert deflections[0, 1] == pytest.approx(0.4472135 * np.sqrt(5), 1e-3)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[-1.0, -1.0]]))
        assert deflections[0, 0] == pytest.approx(-1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(-1.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(1.0, 2.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 3.0]]))
        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(1.0, 2.0), kappa=-1.0)

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[2.0, 3.0]]))
        assert deflections[0, 0] == pytest.approx(-1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(-1.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(
            grid=ag.Grid2D.manual_native(
                [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
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

        deflections = mass_sheet.deflections_from_grid(grid=np.array([[1.0, 0.0]]))

        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.0, 1e-3)

    def test__deflections__change_geometry(self):
        mass_sheet_0 = ag.mp.MassSheet(centre=(0.0, 0.0))
        mass_sheet_1 = ag.mp.MassSheet(centre=(1.0, 1.0))
        deflections_0 = mass_sheet_0.deflections_from_grid(grid=np.array([[1.0, 1.0]]))
        deflections_1 = mass_sheet_1.deflections_from_grid(grid=np.array([[0.0, 0.0]]))
        assert deflections_0[0, 0] == pytest.approx(-deflections_1[0, 0], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(-deflections_1[0, 1], 1e-5)

        mass_sheet_0 = ag.mp.MassSheet(centre=(0.0, 0.0))
        mass_sheet_1 = ag.mp.MassSheet(centre=(0.0, 0.0))
        deflections_0 = mass_sheet_0.deflections_from_grid(grid=np.array([[1.0, 0.0]]))
        deflections_1 = mass_sheet_1.deflections_from_grid(grid=np.array([[0.0, 1.0]]))
        assert deflections_0[0, 0] == pytest.approx(deflections_1[0, 1], 1e-5)
        assert deflections_0[0, 1] == pytest.approx(deflections_1[0, 0], 1e-5)

    def test__multiple_coordinates_in__multiple_coordinates_out(self):
        mass_sheet = ag.mp.MassSheet(centre=(1.0, 2.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(
            grid=np.array([[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]])
        )

        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)
        assert deflections[1, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[1, 1] == pytest.approx(1.0, 1e-3)
        assert deflections[2, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[2, 1] == pytest.approx(1.0, 1e-3)

        mass_sheet = ag.mp.MassSheet(centre=(0.0, 0.0), kappa=1.0)

        deflections = mass_sheet.deflections_from_grid(
            grid=np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 1.0], [2.0, 2.0]])
        )

        assert deflections[0, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[0, 1] == pytest.approx(1.0, 1e-3)

        assert deflections[1, 0] == pytest.approx(2.0, 1e-3)
        assert deflections[1, 1] == pytest.approx(2.0, 1e-3)

        assert deflections[2, 0] == pytest.approx(1.0, 1e-3)
        assert deflections[2, 1] == pytest.approx(1.0, 1e-3)

        assert deflections[3, 0] == pytest.approx(2.0, 1e-3)
        assert deflections[3, 1] == pytest.approx(2.0, 1e-3)

    def test__outputs_are_autoarrays(self):

        grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)

        mass_sheet = ag.mp.MassSheet()

        convergence = mass_sheet.convergence_from_grid(grid=grid)

        assert convergence.shape_native == (2, 2)

        potential = mass_sheet.potential_from_grid(grid=grid)

        assert potential.shape_native == (2, 2)

        deflections = mass_sheet.deflections_from_grid(grid=grid)

        assert deflections.shape_native == (2, 2)


class TestExternalShear:
    def test__convergence_returns_zeros(self):

        shear = ag.mp.ExternalShear(elliptical_comps=(0.1, 0.0))
        convergence = shear.convergence_from_grid(grid=np.array([[0.1, 0.1]]))
        assert (convergence == np.array([0.0])).all()

        shear = ag.mp.ExternalShear(elliptical_comps=(0.1, 0.0))
        convergence = shear.convergence_from_grid(
            grid=np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        )
        assert (convergence == np.array([0.0, 0.0, 0.0])).all()

        convergence = shear.convergence_from_grid(
            grid=ag.Grid2D.manual_native(
                [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
                sub_size=2,
                pixel_scales=(1.0, 1.0),
            )
        )

        assert convergence[0] == pytest.approx(0.0, 1e-3)
        assert convergence[1] == pytest.approx(0.0, 1e-3)
        assert convergence[2] == pytest.approx(0.0, 1e-3)
        assert convergence[3] == pytest.approx(0.0, 1e-3)

        convergence = shear.convergence_from_grid(grid=np.array([[1.0, 0.0]]))

        assert convergence[0] == pytest.approx(0.0, 1e-3)

    def test__potential_returns_zeros(self):
        shear = ag.mp.ExternalShear(elliptical_comps=(0.1, 0.0))
        potential = shear.potential_from_grid(grid=np.array([[0.1, 0.1]]))
        assert (potential == np.array([[0.0, 0.0]])).all()

        shear = ag.mp.ExternalShear(elliptical_comps=(0.1, 0.0))
        potential = shear.potential_from_grid(
            grid=np.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])
        )
        assert (potential == np.array([0.0, 0.0, 0.0])).all()

        potential = shear.potential_from_grid(
            grid=ag.Grid2D.manual_native(
                [[[1.0, 0.0], [1.0, 0.0]], [[1.0, 0.0], [1.0, 0.0]]],
                sub_size=2,
                pixel_scales=(1.0, 1.0),
            )
        )

        assert potential[0] == pytest.approx(0.0, 1e-3)
        assert potential[1] == pytest.approx(0.0, 1e-3)
        assert potential[2] == pytest.approx(0.0, 1e-3)
        assert potential[3] == pytest.approx(0.0, 1e-3)

        potential = shear.potential_from_grid(grid=np.array([[1.0, 0.0]]))

        assert potential[0] == pytest.approx(0.0, 1e-3)

    def test__deflections_correct_values(self):

        shear = ag.mp.ExternalShear(elliptical_comps=(0.1, 0.0))
        deflections = shear.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(0.01625, 1e-3)
        assert deflections[0, 1] == pytest.approx(0.01625, 1e-3)

        shear = ag.mp.ExternalShear(elliptical_comps=(0.1, -0.17320))
        deflections = shear.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))
        assert deflections[0, 0] == pytest.approx(0.04439, 1e-3)
        assert deflections[0, 1] == pytest.approx(-0.011895, 1e-3)

        deflections = shear.deflections_from_grid(
            grid=ag.Grid2D.manual_native(
                [
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

        deflections = shear.deflections_from_grid(grid=np.array([[0.1625, 0.1625]]))

        assert deflections[0, 0] == pytest.approx(0.04439, 1e-3)
        assert deflections[0, 1] == pytest.approx(-0.011895, 1e-3)

    def test__outputs_are_autoarrays(self):

        grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)

        shear = ag.mp.ExternalShear()

        convergence = shear.convergence_from_grid(grid=grid)

        assert convergence.shape_native == (2, 2)

        potential = shear.potential_from_grid(grid=grid)

        assert potential.shape_native == (2, 2)

        deflections = shear.deflections_from_grid(grid=grid)

        assert deflections.shape_native == (2, 2)


class TestInputDeflections:
    def test__deflections_from_grid__grid_coordinates_overlap_image_grid_of_deflections(
        self,
    ):

        deflections_y = ag.Array2D.manual_native(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            pixel_scales=0.1,
            origin=(0.0, 0.0),
        )
        deflections_x = ag.Array2D.manual_native(
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
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

        deflections = input_deflections.deflections_from_grid(grid=grid)

        assert deflections[:, 0] == pytest.approx(deflections_y, 1.0e-4)
        assert deflections[:, 1] == pytest.approx(deflections_x, 1.0e-4)

        grid = ag.Grid2D.manual_slim(
            grid=np.array([[0.1, 0.0], [0.0, 0.0], [-0.1, -0.1]]),
            shape_native=deflections_y.shape_native,
            pixel_scales=deflections_y.pixel_scales,
        )

        deflections = input_deflections.deflections_from_grid(grid=grid)

        assert deflections[:, 0] == pytest.approx([2.0, 5.0, 7.0], 1.0e-4)
        assert deflections[:, 1] == pytest.approx([8.0, 5.0, 3.0], 1.0e-4)

        # input_deflections = ag.mp.InputDeflections(
        #     deflections_y=deflections_y,
        #     deflections_x=deflections_x,
        #     image_plane_grid=image_plane_grid,
        #     normalization_scale=2.0,
        # )
        #
        # deflections = input_deflections.deflections_from_grid(grid=grid)
        #
        # assert deflections[:, 0] == pytest.approx([4.0, 10.0, 14.0], 1.0e-4)
        # assert deflections[:, 1] == pytest.approx([16.0, 10.0, 6.0], 1.0e-4)

    def test__deflections_from_grid__grid_coordinates_dont_overlap_image_grid_of_deflections__uses_interpolation(
        self,
    ):

        deflections_y = ag.Array2D.manual_native(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            pixel_scales=0.1,
            origin=(0.0, 0.0),
        )
        deflections_x = ag.Array2D.manual_native(
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
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

        grid = ag.Grid2D.manual_slim(
            grid=np.array([[0.05, 0.03], [0.02, 0.01], [-0.08, -0.04]]),
            shape_native=deflections_y.shape_native,
            pixel_scales=deflections_y.pixel_scales,
        )

        deflections = input_deflections.deflections_from_grid(grid=grid)

        assert deflections[:, 0] == pytest.approx([3.8, 4.5, 7.0], 1.0e-4)
        assert deflections[:, 1] == pytest.approx([6.2, 5.5, 3.0], 1.0e-4)

    def test__deflections_from_grid__preload_grid_deflections_used_if_preload_grid_input(
        self,
    ):

        deflections_y = ag.Array2D.manual_native(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            pixel_scales=0.1,
            origin=(0.0, 0.0),
        )
        deflections_x = ag.Array2D.manual_native(
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
            pixel_scales=0.1,
            origin=(0.0, 0.0),
        )

        image_plane_grid = ag.Grid2D.uniform(
            shape_native=deflections_y.shape_native,
            pixel_scales=deflections_y.pixel_scales,
        )

        grid = ag.Grid2D.manual_slim(
            grid=np.array([[0.05, 0.03], [0.02, 0.01], [-0.08, -0.04]]),
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

        deflections = input_deflections.deflections_from_grid(grid=grid)

        assert (deflections == input_deflections.preload_deflections).all()

        # input_deflections = ag.mp.InputDeflections(
        #     deflections_y=deflections_y,
        #     deflections_x=deflections_x,
        #     image_plane_grid=image_plane_grid,
        #     preload_grid=grid,
        #     normalization_scale=2.0,
        # )
        #
        # input_deflections.preload_deflections[0, 0] = 1.0
        #
        # deflections = input_deflections.deflections_from_grid(grid=grid)
        #
        # assert (deflections == 2.0 * input_deflections.preload_deflections).all()

    def test__deflections_from_grid__input_grid_extends_beyond_image_plane_grid__raises_exception(
        self,
    ):

        deflections_y = ag.Array2D.manual_native(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            pixel_scales=0.1,
            origin=(0.0, 0.0),
        )
        deflections_x = ag.Array2D.manual_native(
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
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

        grid = ag.Grid2D.manual_slim(
            grid=np.array([[0.0999, 0.0]]),
            shape_native=deflections_y.shape_native,
            pixel_scales=deflections_y.pixel_scales,
        )
        input_deflections.deflections_from_grid(grid=grid)

        grid = ag.Grid2D.manual_slim(
            grid=np.array([[0.0, 0.0999]]),
            shape_native=deflections_y.shape_native,
            pixel_scales=deflections_y.pixel_scales,
        )
        input_deflections.deflections_from_grid(grid=grid)

        with pytest.raises(exc.ProfileException):
            grid = ag.Grid2D.manual_slim(
                grid=np.array([[0.11, 0.0]]),
                shape_native=deflections_y.shape_native,
                pixel_scales=deflections_y.pixel_scales,
            )
            input_deflections.deflections_from_grid(grid=grid)

            with pytest.raises(exc.ProfileException):
                grid = ag.Grid2D.manual_slim(
                    grid=np.array([[0.0, 0.11]]),
                    shape_native=deflections_y.shape_native,
                    pixel_scales=deflections_y.pixel_scales,
                )
                input_deflections.deflections_from_grid(grid=grid)

    def test__convergence_from_grid_potential_from_grid(self):

        deflections_y = ag.Array2D.manual_native(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            pixel_scales=0.1,
            origin=(0.0, 0.0),
        )
        deflections_x = ag.Array2D.manual_native(
            [[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]],
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

        convergence = input_deflections.convergence_from_grid(grid=grid)

        convergence_via_jacobian = input_deflections.convergence_via_jacobian_from_grid(
            grid=grid
        )

        assert (convergence == convergence_via_jacobian).all()

        potential = input_deflections.potential_from_grid(grid=grid)

        assert (potential == np.zeros(shape=(9,))).all()
