import numpy as np

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__mass_quantity_functions__output_is_autoarray_structure():

    grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=1.0, sub_size=1)

    point_mass = ag.mp.PointMass()
    deflections = point_mass.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    cored_power_law = ag.mp.EllPowerLawBroken()

    convergence = cored_power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    deflections = cored_power_law.deflections_yx_2d_from(grid=grid)
    assert deflections.shape_native == (2, 2)

    cored_power_law = ag.mp.SphPowerLawBroken()

    convergence = cored_power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    deflections = cored_power_law.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    cored_power_law = ag.mp.EllPowerLawCored()

    convergence = cored_power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = cored_power_law.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = cored_power_law.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    cored_power_law = ag.mp.SphPowerLawCored()

    convergence = cored_power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = cored_power_law.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = cored_power_law.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    power_law = ag.mp.EllPowerLaw()

    convergence = power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = power_law.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = power_law.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    power_law = ag.mp.SphPowerLaw()

    convergence = power_law.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = power_law.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = power_law.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    cored_isothermal = ag.mp.EllIsothermalCored()

    convergence = cored_isothermal.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = cored_isothermal.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = cored_isothermal.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    cored_isothermal = ag.mp.SphIsothermalCored()

    convergence = cored_isothermal.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = cored_isothermal.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = cored_isothermal.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    isothermal = ag.mp.EllIsothermal()

    convergence = isothermal.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = isothermal.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = isothermal.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)

    isothermal = ag.mp.SphIsothermal()

    convergence = isothermal.convergence_2d_from(grid=grid)
    assert convergence.shape_native == (2, 2)

    potential = isothermal.potential_2d_from(grid=grid)
    assert potential.shape_native == (2, 2)

    deflections = isothermal.deflections_yx_2d_from(grid=grid)
    assert isinstance(deflections, ag.VectorYX2D)
    assert deflections.shape_native == (2, 2)
