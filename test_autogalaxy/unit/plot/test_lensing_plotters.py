import os
import shutil
from os import path

from autoconf import conf
import autogalaxy as ag
import autogalaxy.plot as aplt
import numpy as np
import pytest

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plotter_setup():
    return "{}/files/plots/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "files/plotter"), path.join(directory, "output")
    )


class TestLensingPlotterAttributes:
    def test__light_profile_centres_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.light_profile_centres_scatterer.size == 10
        assert plotter.light_profile_centres_scatterer.marker == "+"
        assert plotter.light_profile_centres_scatterer.colors == ["k", "r"]

        plotter = aplt.Plotter(
            light_profile_centres_scatterer=aplt.LightProfileCentreScatterer(
                size=1, marker=".", colors="k"
            )
        )

        assert plotter.light_profile_centres_scatterer.size == 1
        assert plotter.light_profile_centres_scatterer.marker == "."
        assert plotter.light_profile_centres_scatterer.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.light_profile_centres_scatterer.size == 15
        assert sub_plotter.light_profile_centres_scatterer.marker == "."
        assert sub_plotter.light_profile_centres_scatterer.colors == ["b"]

        sub_plotter = aplt.SubPlotter(
            light_profile_centres_scatterer=aplt.LightProfileCentreScatterer.sub(
                marker="o", colors="r"
            )
        )

        assert sub_plotter.light_profile_centres_scatterer.size == 15
        assert sub_plotter.light_profile_centres_scatterer.marker == "o"
        assert sub_plotter.light_profile_centres_scatterer.colors == ["r"]

    def test__mass_profile_centres_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.mass_profile_centres_scatterer.size == 11
        assert plotter.mass_profile_centres_scatterer.marker == "x"
        assert plotter.mass_profile_centres_scatterer.colors == ["r", "k"]

        plotter = aplt.Plotter(
            mass_profile_centres_scatterer=aplt.MassProfileCentreScatterer(
                size=1, marker=".", colors="k"
            )
        )

        assert plotter.mass_profile_centres_scatterer.size == 1
        assert plotter.mass_profile_centres_scatterer.marker == "."
        assert plotter.mass_profile_centres_scatterer.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.mass_profile_centres_scatterer.size == 16
        assert sub_plotter.mass_profile_centres_scatterer.marker == "o"
        assert sub_plotter.mass_profile_centres_scatterer.colors == ["k"]

        sub_plotter = aplt.SubPlotter(
            mass_profile_centres_scatterer=aplt.MassProfileCentreScatterer.sub(
                marker="o", colors="r"
            )
        )

        assert sub_plotter.mass_profile_centres_scatterer.size == 16
        assert sub_plotter.mass_profile_centres_scatterer.marker == "o"
        assert sub_plotter.mass_profile_centres_scatterer.colors == ["r"]

    def test__multiple_images_scatterer__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.multiple_images_scatterer.size == 12
        assert plotter.multiple_images_scatterer.marker == "o"
        assert plotter.multiple_images_scatterer.colors == ["k", "w"]

        plotter = aplt.Plotter(
            multiple_images_scatterer=aplt.MultipleImagesScatterer(
                size=1, marker=".", colors="k"
            )
        )

        assert plotter.multiple_images_scatterer.size == 1
        assert plotter.multiple_images_scatterer.marker == "."
        assert plotter.multiple_images_scatterer.colors == ["k"]

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.multiple_images_scatterer.size == 17
        assert sub_plotter.multiple_images_scatterer.marker == "."
        assert sub_plotter.multiple_images_scatterer.colors == ["g"]

        sub_plotter = aplt.SubPlotter(
            multiple_images_scatterer=aplt.MultipleImagesScatterer.sub(
                marker="o", colors="r"
            )
        )

        assert sub_plotter.multiple_images_scatterer.size == 17
        assert sub_plotter.multiple_images_scatterer.marker == "o"
        assert sub_plotter.multiple_images_scatterer.colors == ["r"]

    def test__critical_curves_liner__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.critical_curves_liner.width == 2
        assert plotter.critical_curves_liner.style == "-"
        assert plotter.critical_curves_liner.colors == ["w", "k"]
        assert plotter.critical_curves_liner.pointsize == 20

        plotter = aplt.Plotter(
            critical_curves_liner=aplt.CriticalCurvesLiner(
                width=1, style=".", colors="k", pointsize=3
            )
        )

        assert plotter.critical_curves_liner.width == 1
        assert plotter.critical_curves_liner.style == "."
        assert plotter.critical_curves_liner.colors == ["k"]
        assert plotter.critical_curves_liner.pointsize == 3

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.critical_curves_liner.width == 6
        assert sub_plotter.critical_curves_liner.style == "-"
        assert sub_plotter.critical_curves_liner.colors == ["b"]
        assert sub_plotter.critical_curves_liner.pointsize == 22

        sub_plotter = aplt.SubPlotter(
            critical_curves_liner=aplt.CriticalCurvesLiner.sub(
                style=".", colors="r", pointsize=21
            )
        )

        assert sub_plotter.critical_curves_liner.width == 6
        assert sub_plotter.critical_curves_liner.style == "."
        assert sub_plotter.critical_curves_liner.colors == ["r"]
        assert sub_plotter.critical_curves_liner.pointsize == 21

    def test__caustics_liner__from_config_or_via_manual_input(self):

        plotter = aplt.Plotter()

        assert plotter.caustics_liner.width == 3
        assert plotter.caustics_liner.style == "--"
        assert plotter.caustics_liner.colors == ["w", "g"]
        assert plotter.caustics_liner.pointsize == 21

        plotter = aplt.Plotter(
            caustics_liner=aplt.CausticsLiner(
                width=1, style=".", colors="k", pointsize=3
            )
        )

        assert plotter.caustics_liner.width == 1
        assert plotter.caustics_liner.style == "."
        assert plotter.caustics_liner.colors == ["k"]
        assert plotter.caustics_liner.pointsize == 3

        sub_plotter = aplt.SubPlotter()

        assert sub_plotter.caustics_liner.width == 7
        assert sub_plotter.caustics_liner.style == "--"
        assert sub_plotter.caustics_liner.colors == ["g"]
        assert sub_plotter.caustics_liner.pointsize == 23

        sub_plotter = aplt.SubPlotter(
            caustics_liner=aplt.CausticsLiner.sub(style=".", colors="r", pointsize=21)
        )

        assert sub_plotter.caustics_liner.width == 7
        assert sub_plotter.caustics_liner.style == "."
        assert sub_plotter.caustics_liner.colors == ["r"]
        assert sub_plotter.caustics_liner.pointsize == 21


class TestLensingPlotterPlots:
    def test__plot_array__works_with_all_extras_included(self, plot_path, plot_patch):

        array = ag.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        mask = ag.Mask.circular(
            shape_2d=array.shape_2d,
            pixel_scales=array.pixel_scales,
            radius=5.0,
            centre=(2.0, 2.0),
        )

        grid = ag.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.5)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array1", format="png")
        )

        plotter.plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=ag.GridCoordinates([(-1.0, -1.0)]),
            array_overlay=array,
            light_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            mass_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            multiple_images=ag.GridCoordinates([(1.0, 1.0)]),
            critical_curves=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            include_origin=True,
            include_border=True,
        )

        assert plot_path + "array1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array2", format="png")
        )

        plotter.plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=ag.GridCoordinates([[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]]),
            critical_curves=ag.GridCoordinates(
                [[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]
            ),
            include_origin=True,
            include_border=True,
        )

        assert plot_path + "array2.png" in plot_patch.paths

        aplt.Array(
            array=array,
            mask=mask,
            grid=grid,
            positions=ag.GridCoordinates([(-1.0, -1.0)]),
            light_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            mass_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            multiple_images=ag.GridCoordinates([(1.0, 1.0)]),
            critical_curves=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="array3", format="png")
            ),
        )

        assert plot_path + "array3.png" in plot_patch.paths

    def test__plot_array__fits_files_output_correctly(self, plot_path):

        plot_path = plot_path + "/fits/"

        if os.path.exists(plot_path):
            shutil.rmtree(plot_path)

        arr = ag.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="array", format="fits")
        )

        plotter.plot_array(array=arr)

        arr = ag.util.array.numpy_array_2d_from_fits(
            file_path=plot_path + "/array.fits", hdu=0
        )

        assert (arr == np.ones(shape=(31, 31))).all()

        mask = ag.Mask.circular(
            shape_2d=(31, 31), pixel_scales=(1.0, 1.0), radius=5.0, centre=(2.0, 2.0)
        )

        masked_array = ag.Array.manual_mask(array=arr, mask=mask)

        plotter.plot_array(array=masked_array)

        arr = ag.util.array.numpy_array_2d_from_fits(
            file_path=plot_path + "/array.fits", hdu=0
        )

        assert arr.shape == (13, 13)

    def test__plot_grid__works_with_all_extras_included(self, plot_path, plot_patch):
        grid = ag.Grid.uniform(shape_2d=(11, 11), pixel_scales=1.0)
        color_array = np.linspace(start=0.0, stop=1.0, num=grid.shape_1d)

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="grid1", format="png")
        )

        plotter.plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            light_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            mass_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            multiple_images=ag.GridCoordinates([(1.0, 1.0)]),
            critical_curves=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=False,
        )

        assert plot_path + "grid1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="grid2", format="png")
        )

        plotter.plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            light_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            mass_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            multiple_images=ag.GridCoordinates([(1.0, 1.0)]),
            critical_curves=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
        )

        assert plot_path + "grid2.png" in plot_patch.paths

        aplt.Grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            light_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            mass_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            multiple_images=ag.GridCoordinates([(1.0, 1.0)]),
            critical_curves=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="grid3", format="png")
            ),
        )

        assert plot_path + "grid3.png" in plot_patch.paths

    def test__plot_line__works_with_all_extras_included(self, plot_path, plot_patch):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="line1", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert plot_path + "line1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="line2", format="png")
        )

        plotter.plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="semilogy",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert plot_path + "line2.png" in plot_patch.paths

        aplt.Line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="line3", format="png")
            ),
        )

        assert plot_path + "line3.png" in plot_patch.paths

    def test__plot_rectangular_mapper__works_with_all_extras_included(
        self, rectangular_mapper_7x7_3x3, plot_path, plot_patch
    ):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper1", format="png")
        )

        plotter.plot_rectangular_mapper(
            mapper=rectangular_mapper_7x7_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            light_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            mass_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            multiple_images=ag.GridCoordinates([(1.0, 1.0)]),
            critical_curves=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert plot_path + "mapper1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper2", format="png")
        )

        plotter.plot_rectangular_mapper(
            mapper=rectangular_mapper_7x7_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            light_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            mass_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            multiple_images=ag.GridCoordinates([(1.0, 1.0)]),
            critical_curves=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert plot_path + "mapper2.png" in plot_patch.paths

        aplt.MapperObj(
            mapper=rectangular_mapper_7x7_3x3,
            light_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            mass_profile_centres=ag.GridCoordinates([(1.0, 1.0)]),
            multiple_images=ag.GridCoordinates([(1.0, 1.0)]),
            critical_curves=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridCoordinates([(1.0, 1.0), (2.0, 2.0)]),
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert plot_path + "mapper3.png" in plot_patch.paths

    def test__plot_voronoi_mapper__works_with_all_extras_included(
        self, voronoi_mapper_9_3x3, plot_path, plot_patch
    ):

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper1", format="png")
        )

        plotter.plot_voronoi_mapper(
            mapper=voronoi_mapper_9_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert plot_path + "mapper1.png" in plot_patch.paths

        plotter = aplt.Plotter(
            output=aplt.Output(path=plot_path, filename="mapper2", format="png")
        )

        plotter.plot_voronoi_mapper(
            mapper=voronoi_mapper_9_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
        )

        assert plot_path + "mapper2.png" in plot_patch.paths

        aplt.MapperObj(
            mapper=voronoi_mapper_9_3x3,
            image_pixel_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            source_pixel_indexes=[[0, 1], [2]],
            plotter=aplt.Plotter(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert plot_path + "mapper3.png" in plot_patch.paths


class TestInclude:
    def test__critical_curves_from_object(self, lp_0, mp_0):

        include = aplt.Include(critical_curves=False)

        critical_curves = include.critical_curves_from_obj(obj=mp_0)

        assert critical_curves == None

        include = aplt.Include(critical_curves=True)

        critical_curves = include.critical_curves_from_obj(obj=lp_0)

        assert critical_curves == None

        include = aplt.Include(critical_curves=True)

        critical_curves = include.critical_curves_from_obj(obj=mp_0)

        assert critical_curves[0] == pytest.approx(mp_0.critical_curves[0], 1.0e-4)
        assert critical_curves[1] == pytest.approx(mp_0.critical_curves[1], 1.0e-4)

    def test__caustics_from_object(self, lp_0, mp_0):

        include = aplt.Include(caustics=False)

        caustics = include.caustics_from_obj(obj=mp_0)

        assert caustics == None

        include = aplt.Include(caustics=True)

        caustics = include.caustics_from_obj(obj=lp_0)

        assert caustics == None

        include = aplt.Include(caustics=True)

        caustics = include.caustics_from_obj(obj=mp_0)

        assert caustics[0] == pytest.approx(mp_0.caustics[0], 1.0e-4)
        assert caustics[1] == pytest.approx(mp_0.caustics[1], 1.0e-4)

    def test__new_include_with_preloaded_critical_curves_and_caustics(self):

        include = aplt.Include(mask=True)

        assert include.preloaded_critical_curves == None
        assert include.preloaded_caustics == None

        include = include.new_include_with_preloaded_critical_curves_and_caustics(
            preloaded_critical_curves=1, preloaded_caustics=2
        )

        assert include.mask == True
        assert include.preloaded_critical_curves == 1
        assert include.preloaded_caustics == 2
