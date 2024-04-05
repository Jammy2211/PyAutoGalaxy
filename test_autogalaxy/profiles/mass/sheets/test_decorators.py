import autogalaxy as ag
import numpy as np

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__mass_quantity_functions__output_is_autoarray_structure():
    grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0)

    mass_sheet = ag.mp.MassSheet()

    convergence = mass_sheet.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = mass_sheet.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = mass_sheet.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    shear = ag.mp.ExternalShear()

    convergence = shear.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = shear.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = shear.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)
