from autoarray.plot import plotters
from autoarray.structures import arrays
from autogalaxy.plot import lensing_plotters, light_profile_plots, mass_profile_plots


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def image(galaxy, grid, positions=None, include=None, plotter=None):
    """Plot the image (e.g. the datas) of a galaxy, on a grid of (y,x) coordinates.

    Set *autogalaxy.datas.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    galaxy : model.galaxy.ag.Galaxy
        The galaxy whose image are plotted.
    grid : grid_like or datas.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    plotter.plot_array(
        array=galaxy.image_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=galaxy),
        light_profile_centres=include.light_profile_centres_from_obj(obj=galaxy),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=galaxy),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def convergence(galaxy, grid, positions=None, include=None, plotter=None):
    """Plot the convergence of a galaxy, on a grid of (y,x) coordinates.

    Set *autogalaxy.datas.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    galaxy : model.galaxy.ag.Galaxy
        The galaxy whose convergence is plotted.
    grid : grid_like or datas.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    plotter.plot_array(
        array=galaxy.convergence_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=galaxy),
        light_profile_centres=include.light_profile_centres_from_obj(obj=galaxy),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=galaxy),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def potential(galaxy, grid, positions=None, include=None, plotter=None):
    """Plot the potential of a galaxy, on a grid of (y,x) coordinates.

     Set *autogalaxy.datas.arrays.plotters.plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
    galaxy : model.galaxy.ag.Galaxy
         The galaxy whose potential is plotted.
    grid : grid_like or datas.arrays.grid_stacks.Grid
         The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
     """
    plotter.plot_array(
        array=galaxy.potential_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=galaxy),
        light_profile_centres=include.light_profile_centres_from_obj(obj=galaxy),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=galaxy),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def deflections_y(galaxy, grid, positions=None, include=None, plotter=None):
    """Plot the y component of the deflection angles of a galaxy, on a grid of (y,x) coordinates.

    Set *autogalaxy.datas.arrays.plotters.plotters* for a description of all innput parameters not described below.

    Parameters
    -----------
    galaxy : model.galaxy.ag.Galaxy
        The galaxy whose y deflecton angles are plotted.
    grid : grid_like or datas.arrays.grid_stacks.Grid
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    """
    deflections = galaxy.deflections_from_grid(grid=grid)
    deflections_y = arrays.Array.manual_mask(
        array=deflections.in_1d[:, 0], mask=grid.mask
    )

    plotter.plot_array(
        array=deflections_y,
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=galaxy),
        light_profile_centres=include.light_profile_centres_from_obj(obj=galaxy),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=galaxy),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def deflections_x(galaxy, grid, positions=None, include=None, plotter=None):
    """Plot the x component of the deflection angles of a galaxy, on a grid of (y,x) coordinates.

     Set *autogalaxy.datas.arrays.plotters.plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
    galaxy : model.galaxy.ag.Galaxy
         The galaxy whose x deflecton angles are plotted.
     grid : grid_like or datas.arrays.grid_stacks.Grid
         The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
     """
    deflections = galaxy.deflections_from_grid(grid=grid)
    deflections_x = arrays.Array.manual_mask(
        array=deflections.in_1d[:, 1], mask=grid.mask
    )
    plotter.plot_array(
        array=deflections_x,
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=galaxy),
        light_profile_centres=include.light_profile_centres_from_obj(obj=galaxy),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=galaxy),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def magnification(galaxy, grid, positions=None, include=None, plotter=None):
    """Plot the magnification of a galaxy, on a grid of (y,x) coordinates.

     Set *autogalaxy.datas.arrays.plotters.plotters* for a description of all innput parameters not described below.

     Parameters
     -----------
    galaxy : model.galaxy.ag.Galaxy
         The galaxy whose magnification is plotted.
    grid : grid_like or datas.arrays.grid_stacks.Grid
         The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
     """

    plotter.plot_array(
        array=galaxy.magnification_from_grid(grid=grid),
        mask=include.mask_from_grid(grid=grid),
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=galaxy),
        light_profile_centres=include.light_profile_centres_from_obj(obj=galaxy),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=galaxy),
        include_origin=include.origin,
    )


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_labels
def image_subplot(galaxy, grid, positions=None, include=None, sub_plotter=None):

    number_subplots = len(galaxy.light_profiles)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for i, light_profile in enumerate(galaxy.light_profiles):

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=i + 1)

        light_profile_plots.image(
            light_profile=light_profile,
            grid=grid,
            positions=positions,
            include=include,
            plotter=sub_plotter,
        )

    sub_plotter.output.subplot_to_figure()
    sub_plotter.figure.close()


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_labels
def convergence_subplot(galaxy, grid, positions=None, include=None, sub_plotter=None):

    number_subplots = len(galaxy.mass_profiles)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=i + 1)

        mass_profile_plots.convergence(
            mass_profile=mass_profile,
            grid=grid,
            positions=positions,
            include=include,
            plotter=sub_plotter,
        )

    sub_plotter.output.subplot_to_figure()
    sub_plotter.figure.close()


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_labels
def potential_subplot(galaxy, grid, positions=None, include=None, sub_plotter=None):

    number_subplots = len(galaxy.mass_profiles)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=i + 1)

        mass_profile_plots.potential(
            mass_profile=mass_profile,
            grid=grid,
            positions=positions,
            include=include,
            plotter=sub_plotter,
        )

    sub_plotter.output.subplot_to_figure()
    sub_plotter.figure.close()


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_labels
def deflections_y_subplot(galaxy, grid, positions=None, include=None, sub_plotter=None):

    number_subplots = len(galaxy.mass_profiles)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=i + 1)

        mass_profile_plots.deflections_y(
            mass_profile=mass_profile,
            grid=grid,
            positions=positions,
            include=include,
            plotter=sub_plotter,
        )

    sub_plotter.output.subplot_to_figure()
    sub_plotter.figure.close()


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_labels
def deflections_x_subplot(galaxy, grid, positions=None, include=None, sub_plotter=None):

    number_subplots = len(galaxy.mass_profiles)

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    for i, mass_profile in enumerate(galaxy.mass_profiles):

        sub_plotter.setup_subplot(number_subplots=number_subplots, subplot_index=i + 1)

        mass_profile_plots.deflections_x(
            mass_profile=mass_profile,
            grid=grid,
            positions=positions,
            include=include,
            plotter=sub_plotter,
        )

    sub_plotter.output.subplot_to_figure()
    sub_plotter.figure.close()


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def contribution_map(galaxy, mask=None, positions=None, include=None, plotter=None):
    """Plot the summed contribution maps of a hyper_galaxies-fit.

    Set *autogalaxy.datas.arrays.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The hyper_galaxies-fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the contribution_maps are plotted.
    """

    plotter.plot_array(
        array=galaxy.contribution_map,
        mask=mask,
        positions=positions,
        critical_curves=include.critical_curves_from_obj(obj=galaxy),
        light_profile_centres=include.light_profile_centres_from_obj(obj=galaxy),
        mass_profile_centres=include.mass_profile_centres_from_obj(obj=galaxy),
        include_origin=include.origin,
    )
