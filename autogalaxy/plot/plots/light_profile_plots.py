from autoarray.util import plotter_util
from autoarray.plot.mat_wrap import mat_decorators
from autogalaxy.plot.mat_wrap import lensing_plotter, lensing_include, lensing_visuals


@mat_decorators.set_labels
def image(
    light_profile,
    grid,
    positions=None,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the image of a light profile, on a grid of (y,x) coordinates.

    Set *autogalaxy.hyper_galaxies.arrays.plotter_2d.plotter_2d* for a description of all innput parameters not described below.

    Parameters
    -----------
    light_profile : model.profiles.light_profiles.LightProfile
        The light profile whose image are plotted.
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    structure_plots.plot_array(
        array=light_profile.image_from_grid(grid=grid),
        mask=include_2d.mask_from_grid(grid=grid),
        positions=positions,
        light_profile_centres=include_2d.light_profile_centres_from_obj(
            obj=light_profile
        ),
        include_origin=include_2d.origin,
    )


def luminosity_within_circle_in_electrons_per_second_as_function_of_radius(
    light_profile,
    minimum_radius=1.0e-4,
    maximum_radius=10.0,
    radii_bins=10,
    plot_axis_type="semilogy",
    plotter_2d=None,
):

    radii = plotter_util.quantity_radii_from_minimum_and_maximum_radii_and_radii_points(
        minimum_radius=minimum_radius,
        maximum_radius=maximum_radius,
        radii_points=radii_bins,
    )

    luminosities = list(
        map(lambda radius: light_profile.luminosity_within_circle(radius=radius), radii)
    )

    structure_plots.plot_array(
        quantity=luminosities, radii=radii, plot_axis_type=plot_axis_type
    )
