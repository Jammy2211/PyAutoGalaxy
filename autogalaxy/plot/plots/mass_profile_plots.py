from autoarray.structures import arrays
from autoarray.plot.mat_wrap import mat_decorators
from autogalaxy.plot.mat_wrap import lensing_plotter, lensing_include, lensing_visuals


@mat_decorators.set_labels
def convergence(
    mass_profile,
    grid,
    positions=None,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the convergence of a mass profile, on a grid of (y,x) coordinates.

    Set *autogalaxy.hyper_galaxies.arrays.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose convergence is plotted.
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    structure_plots.plot_array(
        array=mass_profile.convergence_from_grid(grid=grid),
        mask=include_2d.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include_2d.critical_curves_from_obj(obj=mass_profile),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def potential(
    mass_profile,
    grid,
    positions=None,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the potential of a mass profile, on a grid of (y,x) coordinates.

    Set *autogalaxy.hyper_galaxies.arrays.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose potential is plotted.
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    structure_plots.plot_array(
        array=mass_profile.potential_from_grid(grid=grid),
        mask=include_2d.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include_2d.critical_curves_from_obj(obj=mass_profile),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def deflections_y(
    mass_profile,
    grid,
    positions=None,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the y component of the deflection angles of a mass profile, on a grid of (y,x) coordinates.

    Set *autogalaxy.hyper_galaxies.arrays.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose y deflecton angles are plotted.
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """

    deflections = mass_profile.deflections_from_grid(grid=grid)
    deflections_y = arrays.Array.manual_mask(
        array=deflections.in_1d[:, 0], mask=grid.mask
    )

    structure_plots.plot_array(
        array=deflections_y,
        mask=include_2d.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include_2d.critical_curves_from_obj(obj=mass_profile),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def deflections_x(
    mass_profile,
    grid,
    positions=None,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the x component of the deflection angles of a mass profile, on a grid of (y,x) coordinates.

    Set *autogalaxy.hyper_galaxies.arrays.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose x deflecton angles are plotted.
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    deflections = mass_profile.deflections_from_grid(grid=grid)
    deflections_x = arrays.Array.manual_mask(
        array=deflections.in_1d[:, 1], mask=grid.mask
    )

    structure_plots.plot_array(
        array=deflections_x,
        mask=include_2d.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include_2d.critical_curves_from_obj(obj=mass_profile),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def magnification(
    mass_profile,
    grid,
    positions=None,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the magnification of a mass profile, on a grid of (y,x) coordinates.

    Set *autogalaxy.hyper_galaxies.arrays.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose magnification is plotted.
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    structure_plots.plot_array(
        array=mass_profile.magnification_from_grid(grid=grid),
        mask=include_2d.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include_2d.critical_curves_from_obj(obj=mass_profile),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include_2d.origin,
    )
