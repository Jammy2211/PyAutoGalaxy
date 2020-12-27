from autoarray.plot.plotter import plotter
from autogalaxy.plot.plotter import lensing_plotter, lensing_include


@lensing_include.set_include
@lensing_plotter.set_plotter_for_figure
@plotter.set_labels
def subplot_image_and_mapper(
    image,
    mapper,
    image_positions=None,
    source_positions=None,
    critical_curves=None,
    caustics=None,
    image_pixel_indexes=None,
    source_pixel_indexes=None,
    include=None,
    plotter=None,
):

    number_subplots = 2

    plotter.open_subplot_figure(number_subplots=number_subplots)

    plotter.setup_subplot(number_subplots=number_subplots, subplot_index=1)

    plotter.plot_array(
        array=image,
        mask=include.mask_from_grid(grid=mapper.grid),
        positions=image_positions,
        lines=critical_curves,
        include_origin=include.origin,
    )

    if image_pixel_indexes is not None:

        plotter.index_scatter.scatter_grid_indexes(
            grid=mapper.grid.mask.geometry.masked_grid, indexes=image_pixel_indexes
        )

    if source_pixel_indexes is not None:

        indexes = mapper.image_pixel_indexes_from_source_pixel_indexes(
            source_pixel_indexes=source_pixel_indexes
        )

        plotter.index_scatter.scatter_grid_indexes(
            grid=mapper.grid.mask.geometry.masked_grid, indexes=indexes
        )

    plotter.setup_subplot(number_subplots=number_subplots, subplot_index=2)

    plotter.plot_mapper(
        mapper=mapper,
        positions=source_positions,
        caustics=caustics,
        image_pixel_indexes=image_pixel_indexes,
        source_pixel_indexes=source_pixel_indexes,
        include_origin=include.origin,
        include_grid=include.inversion_grid,
        include_pixelization_grid=include.inversion_pixelization_grid,
        include_border=include.inversion_border,
    )

    plotter.output.subplot_to_figure()
    plotter.figure.close()
