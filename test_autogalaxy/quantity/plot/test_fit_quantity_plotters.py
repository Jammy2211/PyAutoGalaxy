from os import path

import pytest

import autogalaxy.plot as aplt

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_galaxy_fit_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "plots",
        "galaxy_fitting",
    )


def test__fit_individuals__source_and_galaxy__dependent_on_input(
    fit_quantity_7x7_array_2d,
    fit_quantity_7x7_vector_yx_2d,
    include_2d_all,
    plot_path,
    plot_patch,
):

    fit_quantity_plotter = aplt.FitQuantityPlotter(
        fit=fit_quantity_7x7_array_2d,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_quantity_plotter.figures_2d(
        image=True,
        noise_map=False,
        signal_to_noise_map=False,
        model_image=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "image_2d.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map.png") not in plot_patch.paths
    assert path.join(plot_path, "model_image.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map.png") in plot_patch.paths

    fit_quantity_plotter = aplt.FitQuantityPlotter(
        fit=fit_quantity_7x7_vector_yx_2d,
        include_2d=include_2d_all,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(plot_path, format="png")),
    )

    fit_quantity_plotter.figures_2d(
        image=True,
        noise_map=False,
        signal_to_noise_map=False,
        model_image=True,
        chi_squared_map=True,
    )

    assert path.join(plot_path, "image_2d_y.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_y.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map_y.png") not in plot_patch.paths
    assert path.join(plot_path, "model_image_y.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map_y.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map_y.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map_y.png") in plot_patch.paths

    assert path.join(plot_path, "image_2d_x.png") in plot_patch.paths
    assert path.join(plot_path, "noise_map_x.png") not in plot_patch.paths
    assert path.join(plot_path, "signal_to_noise_map_x.png") not in plot_patch.paths
    assert path.join(plot_path, "model_image_x.png") in plot_patch.paths
    assert path.join(plot_path, "residual_map_x.png") not in plot_patch.paths
    assert path.join(plot_path, "normalized_residual_map_x.png") not in plot_patch.paths
    assert path.join(plot_path, "chi_squared_map_x.png") in plot_patch.paths


def test__fit_sub_plot__all_types_of_fit(
    fit_quantity_7x7_array_2d,
    fit_quantity_7x7_vector_yx_2d,
    include_2d_all,
    plot_patch,
    plot_path,
):

    fit_quantity_plotter = aplt.FitQuantityPlotter(
        fit=fit_quantity_7x7_array_2d,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_quantity_plotter.subplot_fit_quantity()
    assert path.join(plot_path, "subplot_fit_quantity.png") in plot_patch.paths

    fit_quantity_plotter = aplt.FitQuantityPlotter(
        fit=fit_quantity_7x7_vector_yx_2d,
        mat_plot_2d=aplt.MatPlot2D(output=aplt.Output(path=plot_path, format="png")),
    )

    fit_quantity_plotter.subplot_fit_quantity()

    print(plot_patch.paths)

    assert path.join(plot_path, "subplot_fit_quantity_y.png") in plot_patch.paths
    assert path.join(plot_path, "subplot_fit_quantity_x.png") in plot_patch.paths
