from autoarray.plot.mat_wrap import mat_decorators
from autoarray.plot.plots import structure_plots
from autogalaxy.plot.mat_wrap import lensing_plotter, lensing_include, lensing_visuals


def subplot_hyper_images_of_galaxies(
    hyper_galaxy_image_path_dict,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    if hyper_galaxy_image_path_dict is None:
        return

    plotter_2d = plotter_2d.plotter_for_subplot_from(
        func=subplot_hyper_images_of_galaxies
    )

    number_subplots = 0

    for i in hyper_galaxy_image_path_dict.items():
        number_subplots += 1

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    hyper_index = 0

    for path, galaxy_image in hyper_galaxy_image_path_dict.items():

        hyper_index += 1

        plotter_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=hyper_index
        )

        hyper_galaxy_image(
            galaxy_image=galaxy_image,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


def subplot_contribution_maps_of_galaxies(
    contribution_maps_of_galaxies,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):

    plotter_2d = plotter_2d.plotter_for_subplot_from(
        func=subplot_contribution_maps_of_galaxies
    )

    contribution_maps = [
        contribution_map
        for contribution_map in contribution_maps_of_galaxies
        if contribution_map is not None
    ]

    number_subplots = len(contribution_maps)

    if number_subplots == 0:
        return

    plotter_2d.open_subplot_figure(number_subplots=number_subplots)

    hyper_index = 0

    for contribution_map_array in contribution_maps:

        hyper_index += 1

        plotter_2d.setup_subplot(
            number_subplots=number_subplots, subplot_index=hyper_index
        )

        contribution_map(
            contribution_map_in=contribution_map_array,
            plotter_2d=plotter_2d,
            visuals_2d=visuals_2d,
            include_2d=include_2d,
        )

    plotter_2d.output.subplot_to_figure()

    plotter_2d.figure.close()


@mat_decorators.set_labels
def hyper_model_image(
    hyper_model_image,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autogalaxy.datas.arrays.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_galaxy_image : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    structure_plots.plot_array(
        array=hyper_model_image,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )


@mat_decorators.set_labels
def hyper_galaxy_image(
    galaxy_image,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autogalaxy.datas.arrays.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_galaxy_image : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    structure_plots.plot_array(
        array=galaxy_image,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )


@mat_decorators.set_labels
def contribution_map(
    contribution_map_in,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
):
    """Plot the summed contribution maps of a hyper_galaxies-fit.

    Set *autogalaxy.datas.arrays.plotter_2d.plotter_2d* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The hyper_galaxies-fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the contribution_maps are plotted.
    """

    structure_plots.plot_array(
        array=contribution_map_in,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
    )
