from autoarray.plot import plotters
from autogalaxy.plot import lensing_plotters


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_hyper_images_of_galaxies(
    hyper_galaxy_image_path_dict, mask=None, include=None, sub_plotter=None
):

    if hyper_galaxy_image_path_dict is None:
        return

    number_subplots = 0

    for i in hyper_galaxy_image_path_dict.items():
        number_subplots += 1

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    hyper_index = 0

    for path, galaxy_image in hyper_galaxy_image_path_dict.items():

        hyper_index += 1

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=hyper_index
        )

        hyper_galaxy_image(galaxy_image=galaxy_image, mask=mask, plotter=sub_plotter)

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@lensing_plotters.set_include_and_sub_plotter
@plotters.set_subplot_filename
def subplot_contribution_maps_of_galaxies(
    contribution_maps_of_galaxies, mask=None, include=None, sub_plotter=None
):

    contribution_maps = [
        contribution_map
        for contribution_map in contribution_maps_of_galaxies
        if contribution_map is not None
    ]

    number_subplots = len(contribution_maps)

    if number_subplots == 0:
        return

    sub_plotter.open_subplot_figure(number_subplots=number_subplots)

    hyper_index = 0

    for contribution_map_array in contribution_maps:

        hyper_index += 1

        sub_plotter.setup_subplot(
            number_subplots=number_subplots, subplot_index=hyper_index
        )

        contribution_map(
            contribution_map_in=contribution_map_array, mask=mask, plotter=sub_plotter
        )

    sub_plotter.output.subplot_to_figure()

    sub_plotter.figure.close()


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def hyper_model_image(
    hyper_model_image,
    mask=None,
    positions=None,
    image_plane_pix_grid=None,
    include=None,
    plotter=None,
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autogalaxy.datas.arrays.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_galaxy_image : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    plotter.plot_array(
        array=hyper_model_image,
        mask=mask,
        grid=image_plane_pix_grid,
        positions=positions,
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def hyper_galaxy_image(
    galaxy_image,
    mask=None,
    positions=None,
    image_plane_pix_grid=None,
    include=None,
    plotter=None,
):
    """Plot the image of a hyper_galaxies galaxy image.

    Set *autogalaxy.datas.arrays.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    hyper_galaxy_image : datas.imaging.datas.Imaging
        The hyper_galaxies galaxy image.
    origin : True
        If true, the origin of the datas's coordinate system is plotted as a 'x'.
    """

    plotter.plot_array(
        array=galaxy_image, mask=mask, grid=image_plane_pix_grid, positions=positions
    )


@lensing_plotters.set_include_and_plotter
@plotters.set_labels
def contribution_map(
    contribution_map_in, mask=None, positions=None, include=None, plotter=None
):
    """Plot the summed contribution maps of a hyper_galaxies-fit.

    Set *autogalaxy.datas.arrays.plotters.plotters* for a description of all input parameters not described below.

    Parameters
    -----------
    fit : datas.fitting.fitting.AbstractLensHyperFit
        The hyper_galaxies-fit to the datas, which includes a list of every model image, residual_map, chi-squareds, etc.
    image_index : int
        The index of the datas in the datas-set of which the contribution_maps are plotted.
    """

    plotter.plot_array(array=contribution_map_in, mask=mask, positions=positions)
