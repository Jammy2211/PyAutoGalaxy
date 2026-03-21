from os import path

import autogalaxy.plot as aplt
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_mp_plotter_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "profiles"
    )


def test__figures_2d__all_are_output(
    mp_0,
    grid_2d_7x7,
    grid_2d_irregular_7x7_list,
    plot_path,
    plot_patch,
):
    aplt.plot_mass_profile_convergence_2d(
        mass_profile=mp_0,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    aplt.plot_mass_profile_potential_2d(
        mass_profile=mp_0,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    aplt.plot_mass_profile_deflections_y_2d(
        mass_profile=mp_0,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    aplt.plot_mass_profile_deflections_x_2d(
        mass_profile=mp_0,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )
    aplt.plot_mass_profile_magnification_2d(
        mass_profile=mp_0,
        grid=grid_2d_7x7,
        output_path=plot_path,
        output_format="png",
    )

    assert path.join(plot_path, "convergence_2d.png") in plot_patch.paths
    assert path.join(plot_path, "potential_2d.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_y_2d.png") in plot_patch.paths
    assert path.join(plot_path, "deflections_x_2d.png") in plot_patch.paths
    assert path.join(plot_path, "magnification_2d.png") in plot_patch.paths
