import numpy as np

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__mass_quantity_functions__output_is_autoarray_structure():
    grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0)

    point_mass = ag.mp.PointMass()
    deflections = point_mass.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    cored_power_law = ag.mp.PowerLawBroken()

    convergence = cored_power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    deflections = cored_power_law.deflections_yx_2d_from(grid=grid)
    assert deflections.shape_native == (2, 2)

    cored_power_law = ag.mp.PowerLawBrokenSph()

    convergence = cored_power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    deflections = cored_power_law.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    cored_power_law = ag.mp.PowerLawCore()

    convergence = cored_power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = cored_power_law.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = cored_power_law.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    cored_power_law = ag.mp.PowerLawCoreSph()

    convergence = cored_power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = cored_power_law.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = cored_power_law.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    power_law = ag.mp.PowerLaw()

    convergence = power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = power_law.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = power_law.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    power_law = ag.mp.PowerLawSph()

    convergence = power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = power_law.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = power_law.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    cored_isothermal = ag.mp.IsothermalCore()

    convergence = cored_isothermal.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = cored_isothermal.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = cored_isothermal.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    cored_isothermal = ag.mp.IsothermalCoreSph()

    convergence = cored_isothermal.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = cored_isothermal.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = cored_isothermal.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    isothermal = ag.mp.Isothermal()

    convergence = isothermal.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = isothermal.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = isothermal.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    isothermal = ag.mp.IsothermalSph()

    convergence = isothermal.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = isothermal.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = isothermal.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)
