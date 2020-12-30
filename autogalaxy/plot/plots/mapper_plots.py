from autoarray.plot.plots import mapper_plots
from autoarray.plot.mat_wrap import mat_decorators
from autogalaxy.plot.mat_wrap import lensing_plotter, lensing_include, lensing_visuals


@mat_decorators.set_labels
def subplot_image_and_mapper(
    image,
    mapper,
    plotter_2d: lensing_plotter.Plotter2D = lensing_plotter.Plotter2D(),
    visuals_2d: lensing_visuals.Visuals2D = lensing_visuals.Visuals2D(),
    include_2d: lensing_include.Include2D = lensing_include.Include2D(),
    full_indexes=None,
    pixelization_indexes=None,
):

    mapper_plots.subplot_image_and_mapper(
        image=image,
        mapper=mapper,
        plotter_2d=plotter_2d,
        visuals_2d=visuals_2d,
        include_2d=include_2d,
        full_indexes=full_indexes,
        pixelization_indexes=pixelization_indexes,
    )
