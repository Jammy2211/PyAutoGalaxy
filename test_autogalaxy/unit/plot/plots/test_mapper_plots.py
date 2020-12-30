from os import path

import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "mapper"
    )


def test__image_and_mapper_subplot_is_output_for_all_mappers(
    imaging_7x7, rectangular_mapper_7x7_3x3, voronoi_mapper_9_3x3, plot_path, plot_patch
):

    aplt.Mapper.subplot_image_and_mapper(
        image=imaging_7x7.image,
        mapper=rectangular_mapper_7x7_3x3,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
        include_2d=aplt.Include2D(
            mapper_data_pixelization_grid=True,
            mapper_source_pixelization_grid=True,
            mapper_source_full_grid=True,
            mapper_source_border=True,
        ),
        full_indexes=[[0, 1, 2], [3]],
        pixelization_indexes=[[1, 2], [0]],
    )
    assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths

    aplt.Mapper.subplot_image_and_mapper(
        image=imaging_7x7.image,
        mapper=voronoi_mapper_9_3x3,
        plotter_2d=aplt.Plotter2D(output=aplt.Output(path=plot_path, format="png")),
        include_2d=aplt.Include2D(
            mapper_data_pixelization_grid=True,
            mapper_source_pixelization_grid=True,
            mapper_source_full_grid=True,
            mapper_source_border=True,
        ),
        full_indexes=[[0, 1, 2], [3]],
        pixelization_indexes=[[1, 2], [0]],
    )
    assert path.join(plot_path, "subplot_image_and_mapper.png") in plot_patch.paths
