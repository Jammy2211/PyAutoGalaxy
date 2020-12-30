from autoarray.structures import arrays
from autoarray.plot.mat_wrap import mat_decorators
from autogalaxy.plot.mat_wrap import lensing_plotter, lensing_include, lensing_visuals
from autoarray.plot.plots import structure_plots


@mat_decorators.set_labels
def image(
    plane,
    grid,
    positions=None,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    structure_plots.plot_array(
        array=plane.image_from_grid(grid=grid),
        mask=include_2d.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include_2d.critical_curves_from_obj(obj=plane),
        light_profile_centres=include_2d.light_profile_centres_from_obj(obj=plane),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=plane),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def plane_image(
    plane, grid, positions=None, caustics=None, include_2d=None, plotter_2d=None
):

    structure_plots.plot_array(
        array=plane.plane_image_from_grid(grid=grid).array,
        positions=positions,
        caustics=caustics,
        grid=include_2d.grid_from_grid(grid=grid),
        light_profile_centres=include_2d.light_profile_centres_from_obj(obj=plane),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=plane),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def convergence(
    plane,
    grid,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    structure_plots.plot_array(
        array=plane.convergence_from_grid(grid=grid),
        mask=include_2d.mask_from_grid(grid=grid),
        critical_curves=include_2d.critical_curves_from_obj(obj=plane),
        light_profile_centres=include_2d.light_profile_centres_from_obj(obj=plane),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=plane),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def potential(
    plane,
    grid,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    structure_plots.plot_array(
        array=plane.potential_from_grid(grid=grid),
        mask=include_2d.mask_from_grid(grid=grid),
        critical_curves=include_2d.critical_curves_from_obj(obj=plane),
        light_profile_centres=include_2d.light_profile_centres_from_obj(obj=plane),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=plane),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def deflections_y(
    plane,
    grid,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_y = arrays.Array.manual_mask(
        array=deflections.in_1d[:, 0], mask=grid.mask
    )

    structure_plots.plot_array(
        array=deflections_y,
        mask=include_2d.mask_from_grid(grid=grid),
        critical_curves=include_2d.critical_curves_from_obj(obj=plane),
        light_profile_centres=include_2d.light_profile_centres_from_obj(obj=plane),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=plane),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def deflections_x(
    plane,
    grid,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    deflections = plane.deflections_from_grid(grid=grid)
    deflections_x = arrays.Array.manual_mask(
        array=deflections.in_1d[:, 1], mask=grid.mask
    )

    structure_plots.plot_array(
        array=deflections_x,
        mask=include_2d.mask_from_grid(grid=grid),
        critical_curves=include_2d.critical_curves_from_obj(obj=plane),
        light_profile_centres=include_2d.light_profile_centres_from_obj(obj=plane),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=plane),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def magnification(
    plane,
    grid,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    structure_plots.plot_array(
        array=plane.magnification_from_grid(grid=grid),
        mask=include_2d.mask_from_grid(grid=grid),
        critical_curves=include_2d.critical_curves_from_obj(obj=plane),
        light_profile_centres=include_2d.light_profile_centres_from_obj(obj=plane),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=plane),
        include_origin=include_2d.origin,
    )


@mat_decorators.set_labels
def image_and_source_plane_subplot(
    image_plane,
    source_plane,
    grid,
    indexes=None,
    positions=None,
    axis_limits=None,
    include_2d=None,
    plotter_2d=None,
):

    number_subplots = 2

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    plane_grid(
        plane=image_plane,
        grid=grid,
        indexes=indexes,
        axis_limits=axis_limits,
        positions=positions,
        critical_curves=include_2d.critical_curves_from_obj(obj=image_plane),
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    source_plane_grid = image_plane.traced_grid_from_grid(grid=grid)

    plotter_2d.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    plane_grid(
        plane=source_plane,
        grid=source_plane_grid,
        indexes=indexes,
        axis_limits=axis_limits,
        positions=positions,
        caustics=include_2d.caustics_from_obj(obj=image_plane),
        include_2d=include_2d,
        plotter_2d=plotter_2d,
    )

    plotter_2d.output.subplot_to_figure()
    plotter_2d.figure.close()


@mat_decorators.set_labels
def plane_grid(
    plane,
    grid,
    indexes=None,
    axis_limits=None,
    positions=None,
    critical_curves=None,
    caustics=None,
    include_2d=None,
    plotter_2d=None,
):

    plotter_2d._plot_grid(
        grid=grid,
        positions=positions,
        axis_limits=axis_limits,
        indexes=indexes,
        critical_curves=critical_curves,
        caustics=caustics,
        light_profile_centres=include_2d.light_profile_centres_from_obj(obj=plane),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=plane),
        include_origin=include_2d.origin,
        include_border=include_2d.border,
    )


@mat_decorators.set_labels
def contribution_map(
    plane, mask=None, positions=None, include_2d=None, plotter_2d=None
):

    structure_plots.plot_array(
        array=plane.contribution_map,
        mask=mask,
        positions=positions,
        light_profile_centres=include_2d.light_profile_centres_from_obj(obj=plane),
        mass_profile_centres=include_2d.mass_profile_centres_from_obj(obj=plane),
        critical_curves=include_2d.critical_curves_from_obj(obj=plane),
        include_origin=include_2d.origin,
        include_border=include_2d.border,
    )
