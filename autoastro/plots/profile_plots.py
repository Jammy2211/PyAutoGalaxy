from autoarray.plotters import plotters, array_plotters, line_plotters
from autoarray.util import plotter_util
from autoastro.plots import lens_plotter_util


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def image(
    light_profile,
    grid,
    mask=None,
    positions=None,
    include=plotters.Include(),
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the image of a light profile, on a grid of (y,x) coordinates.

    Set *autoastro.hyper_galaxies.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    light_profile : model.profiles.light_profiles.LightProfile
        The light profile whose image are plotted.
    grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    array_plotter.plot_array(
        array=light_profile.profile_image_from_grid(grid=grid),
        mask=mask,
        points=positions,
    )


def luminosity_within_circle_in_electrons_per_second_as_function_of_radius(
    light_profile,
    minimum_radius=1.0e-4,
    maximum_radius=10.0,
    radii_bins=10,
    plot_axis_type="semilogy",
    line_plotter=line_plotters.LinePlotter(),
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

    line_plotter.plot_line(
        quantity=luminosities, radii=radii, plot_axis_type=plot_axis_type
    )


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def convergence(
    mass_profile,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the convergence of a mass profile, on a grid of (y,x) coordinates.

    Set *autoastro.hyper_galaxies.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose convergence is plotted.
    grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """

    convergence = mass_profile.convergence_from_grid(grid=grid)

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=mass_profile,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    array_plotter.plot_array(
        array=convergence, mask=mask, points=positions, lines=lines
    )


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def potential(
    mass_profile,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the potential of a mass profile, on a grid of (y,x) coordinates.

    Set *autoastro.hyper_galaxies.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose potential is plotted.
    grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    potential = mass_profile.potential_from_grid(grid=grid)

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=mass_profile,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    array_plotter.plot_array(array=potential, mask=mask, points=positions, lines=lines)


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def deflections_y(
    mass_profile,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the y component of the deflection angles of a mass profile, on a grid of (y,x) coordinates.

    Set *autoastro.hyper_galaxies.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

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

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=mass_profile,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    array_plotter.plot_array(
        array=deflections_y, mask=mask, points=positions, lines=lines
    )


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def deflections_x(
    mass_profile,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the x component of the deflection angles of a mass profile, on a grid of (y,x) coordinates.

     Set *autoastro.hyper_galaxies.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

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

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=mass_profile,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    array_plotter.plot_array(
        array=deflections_x, mask=mask, points=positions, lines=lines
    )


@lens_plotter_util.set_includes
@lens_plotter_util.set_labels_and_unit_conversion
def magnification(
    mass_profile,
    grid,
    mask=None,
    positions=None,
    include_critical_curves=False,
    include_caustics=False,
    array_plotter=array_plotters.ArrayPlotter(),
):
    """Plot the magnification of a mass profile, on a grid of (y,x) coordinates.

    Set *autoastro.hyper_galaxies.arrays.plotters.array_plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    mass_profile : model.profiles.mass_profiles.MassProfile
        The mass profile whose magnification is plotted.
    grid : ndarray or hyper_galaxies.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    magnification = mass_profile.magnification_from_grid(grid=grid)

    lines = lens_plotter_util.critical_curves_and_caustics_from_lensing_object(
        obj=mass_profile,
        include_critical_curves=include_critical_curves,
        include_caustics=include_caustics,
    )

    array_plotter.plot_array(
        array=magnification, mask=mask, points=positions, lines=lines
    )
