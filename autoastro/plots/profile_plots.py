from autoarray.plotters import plotters
from autoastro.plots import lensing_plotters
from autoarray.util import plotter_util


@plotters.set_labels
def image(
    light_profile,
    grid,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the image of a light profile, on a grid of (y,x) coordinates.

    Set *autoastro.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    light_profile : model.profiles.light_profiles.LightProfile
        The light profile whose image are plotted.
    grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    plotter.array.plot(
        array=light_profile.profile_image_from_grid(grid=grid),
        mask=mask,
        points=positions,
        include_origin=include.origin,
    )


def luminosity_within_circle_in_electrons_per_second_as_function_of_radius(
    light_profile,
    minimum_radius=1.0e-4,
    maximum_radius=10.0,
    radii_bins=10,
    plot_axis_type="semilogy",
        plotter=plotters.Plotter(),
):

    radii = plotter_util.quantity_radii_from_minimum_and_maximum_radii_and_radii_points(
        minimum_radius=minimum_radius,
        maximum_radius=maximum_radius,
        radii_points=radii_bins,
    )

    luminosities = list(
        map(
            lambda radius: light_profile.luminosity_within_circle_in_units(
                radius=radius
            ),
            radii,
        )
    )

    plotter.array.plot(
        quantity=luminosities, radii=radii, plot_axis_type=plot_axis_type
    )


@plotters.set_labels
def convergence(
    mass_profile,
    grid,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the convergence of a mass profile, on a grid of (y,x) coordinates.

    Set *autoastro.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose convergence is plotted.
    grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    plotter.array.plot(
        array=mass_profile.convergence_from_grid(grid=grid),
        mask=mask,
        points=positions,
        lines=include.critical_curves_from_obj(obj=mass_profile),
        centres=include.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include.origin,
    )


@plotters.set_labels
def potential(
    mass_profile,
    grid,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the potential of a mass profile, on a grid of (y,x) coordinates.

    Set *autoastro.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose potential is plotted.
    grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    plotter.array.plot(
        array=mass_profile.potential_from_grid(grid=grid),
        mask=mask,
        points=positions,
        lines=include.critical_curves_from_obj(obj=mass_profile),
        centres=include.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include.origin,
    )


@plotters.set_labels
def deflections_y(
    mass_profile,
    grid,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the y component of the deflection angles of a mass profile, on a grid of (y,x) coordinates.

    Set *autoastro.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose y deflecton angles are plotted.
    grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """

    deflections = mass_profile.deflections_from_grid(grid=grid)
    deflections_y = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 0]
    )

    plotter.array.plot(
        array=deflections_y,
        mask=mask,
        points=positions,
        lines=include.critical_curves_from_obj(obj=mass_profile),
        centres=include.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include.origin,
    )


@plotters.set_labels
def deflections_x(
    mass_profile,
    grid,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the x component of the deflection angles of a mass profile, on a grid of (y,x) coordinates.

     Set *autoastro.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
     mass_profile : model.profiles.mass_profiles.MassProfile
         The mass profile whose x deflecton angles are plotted.
     grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
         The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
     """
    deflections = mass_profile.deflections_from_grid(grid=grid)
    deflections_x = grid.mapping.array_stored_1d_from_sub_array_1d(
        sub_array_1d=deflections[:, 1]
    )

    plotter.array.plot(
        array=deflections_x,
        mask=mask,
        points=positions,
        lines=include.critical_curves_from_obj(obj=mass_profile),
        centres=include.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include.origin,
    )


@plotters.set_labels
def magnification(
    mass_profile,
    grid,
    mask=None,
    positions=None,
    include=lensing_plotters.Include(),
    plotter=plotters.Plotter(),
):
    """Plot the magnification of a mass profile, on a grid of (y,x) coordinates.

    Set *autoastro.hyper_galaxies.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose magnification is plotted.
    grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    plotter.array.plot(
        array=mass_profile.magnification_from_grid(grid=grid),
        mask=mask,
        points=positions,
        lines=include.critical_curves_from_obj(obj=mass_profile),
        centres=include.mass_profile_centres_from_obj(obj=mass_profile),
        include_origin=include.origin,
    )
