from autoarray.plot import plotters
from autoarray.structures import arrays
from autogalaxy.plot import lensing_plotters


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def convergence(mass_profile, grid, positions=None, include=None, plotter=None):
    """Plot the convergence of a mass profile, on a grid of (y,x) coordinates.

    Set *autogalaxy.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose convergence is plotted.
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    plotter.plot_array(
        array=mass_profile.convergence_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=mass_profile),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def potential(mass_profile, grid, positions=None, include=None, plotter=None):
    """Plot the potential of a mass profile, on a grid of (y,x) coordinates.

    Set *autogalaxy.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose potential is plotted.
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    plotter.plot_array(
        array=mass_profile.potential_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=mass_profile),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def deflections_y(mass_profile, grid, positions=None, include=None, plotter=None):
    """Plot the y component of the deflection angles of a mass profile, on a grid of (y,x) coordinates.

    Set *autogalaxy.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

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

    plotter.plot_array(
        array=deflections_y,
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=mass_profile),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def deflections_x(mass_profile, grid, positions=None, include=None, plotter=None):
    """Plot the x component of the deflection angles of a mass profile, on a grid of (y,x) coordinates.

     Set *autogalaxy.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

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

    plotter.plot_array(
        array=deflections_x,
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=mass_profile),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def magnification(mass_profile, grid, positions=None, include=None, plotter=None):
    """Plot the magnification of a mass profile, on a grid of (y,x) coordinates.

    Set *autogalaxy.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose magnification is plotted.
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    plotter.plot_array(
        array=mass_profile.magnification_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=mass_profile),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include.origin,
    )
