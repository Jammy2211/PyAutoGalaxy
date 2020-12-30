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
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots"
    )


class TestLensingPlotterPlots:
    def test__plot_array__works_with_all_extras_included(self, plot_path, plot_patch):

        array = ag.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        mask = ag.Mask2D.circular(
            shape_2d=array.shape_2d,
            pixel_scales=array.pixel_scales,
            radius=5.0,
            centre=(2.0, 2.0),
        )

        grid = ag.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.5)

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="array1", format="png")
        )

        plotter._plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=ag.GridIrregularGrouped([(-1.0, -1.0)]),
            array_overlay=array,
            light_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            mass_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            multiple_images=ag.GridIrregularGrouped([(1.0, 1.0)]),
            critical_curves=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            include_origin=True,
            include_border=True,
        )

        assert path.join(plot_path, "array1.png") in plot_patch.paths

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="array2", format="png")
        )

        plotter._plot_array(
            array=array,
            mask=mask,
            grid=grid,
            positions=ag.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(-1.0, -1.0)]]
            ),
            critical_curves=ag.GridIrregularGrouped(
                [[(1.0, 1.0), (2.0, 2.0)], [(2.0, 4.0), (5.0, 6.0)]]
            ),
            include_origin=True,
            include_border=True,
        )

        assert path.join(plot_path, "array2.png") in plot_patch.paths

        aplt.Array(
            array=array,
            mask=mask,
            grid=grid,
            positions=ag.GridIrregularGrouped([(-1.0, -1.0)]),
            light_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            mass_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            multiple_images=ag.GridIrregularGrouped([(1.0, 1.0)]),
            critical_curves=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            plotter=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="array3", format="png")
            ),
        )

        assert path.join(plot_path, "array3.png") in plot_patch.paths

    def test__plot_array__fits_files_output_correctly(self, plot_path):

        plot_path = path.join(plot_path, "fits")

        if path.exists(plot_path):
            shutil.rmtree(plot_path)

        arr = ag.Array.ones(shape_2d=(31, 31), pixel_scales=(1.0, 1.0), sub_size=2)

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="array", format="fits")
        )

        plotter._plot_array(array=arr)

        arr = ag.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "array.fits"), hdu=0
        )

        assert (arr == np.ones(shape=(31, 31))).all()

        mask = ag.Mask2D.circular(
            shape_2d=(31, 31), pixel_scales=(1.0, 1.0), radius=5.0, centre=(2.0, 2.0)
        )

        masked_array = ag.Array.manual_mask(array=arr, mask=mask)

        plotter._plot_array(array=masked_array)

        arr = ag.util.array.numpy_array_2d_from_fits(
            file_path=path.join(plot_path, "array.fits"), hdu=0
        )

        assert arr.shape == (13, 13)

    def test__plot_grid__works_with_all_extras_included(self, plot_path, plot_patch):
        grid = ag.Grid.uniform(shape_2d=(11, 11), pixel_scales=1.0)
        color_array = np.linspace(start=0.0, stop=1.0, num=grid.shape_1d)

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="grid1", format="png")
        )

        plotter._plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            light_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            mass_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            multiple_images=ag.GridIrregularGrouped([(1.0, 1.0)]),
            critical_curves=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=False,
        )

        assert path.join(plot_path, "grid1.png") in plot_patch.paths

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="grid2", format="png")
        )

        plotter._plot_grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            light_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            mass_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            multiple_images=ag.GridIrregularGrouped([(1.0, 1.0)]),
            critical_curves=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
        )

        assert path.join(plot_path, "grid2.png") in plot_patch.paths

        aplt.Grid(
            grid=grid,
            color_array=color_array,
            axis_limits=[-1.5, 1.5, -2.5, 2.5],
            light_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            mass_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            multiple_images=ag.GridIrregularGrouped([(1.0, 1.0)]),
            critical_curves=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            indexes=[0, 1, 2, 14],
            symmetric_around_centre=True,
            plotter=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="grid3", format="png")
            ),
        )

        assert path.join(plot_path, "grid3.png") in plot_patch.paths

    def test__plot_line__works_with_all_extras_included(self, plot_path, plot_patch):

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="line1", format="png")
        )

        plotter._plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert path.join(plot_path, "line1.png") in plot_patch.paths

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="line2", format="png")
        )

        plotter._plot_line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="semilogy",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
        )

        assert path.join(plot_path, "line2.png") in plot_patch.paths

        aplt.Line(
            y=np.array([1.0, 2.0, 3.0]),
            x=np.array([0.5, 1.0, 1.5]),
            plot_axis_type="loglog",
            vertical_lines=[1.0, 2.0],
            label="line0",
            vertical_line_labels=["line1", "line2"],
            plotter=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="line3", format="png")
            ),
        )

        assert path.join(plot_path, "line3.png") in plot_patch.paths

    def test__plot_rectangular_mapper__works_with_all_extras_included(
        self, rectangular_mapper_7x7_3x3, plot_path, plot_patch
    ):

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="mapper1", format="png")
        )

        plotter._plot_rectangular_mapper(
            mapper=rectangular_mapper_7x7_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            light_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            mass_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            multiple_images=ag.GridIrregularGrouped([(1.0, 1.0)]),
            critical_curves=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="mapper2", format="png")
        )

        plotter._plot_rectangular_mapper(
            mapper=rectangular_mapper_7x7_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            light_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            mass_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            multiple_images=ag.GridIrregularGrouped([(1.0, 1.0)]),
            critical_curves=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        aplt.MapperObj(
            mapper=rectangular_mapper_7x7_3x3,
            light_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            mass_profile_centres=ag.GridIrregularGrouped([(1.0, 1.0)]),
            multiple_images=ag.GridIrregularGrouped([(1.0, 1.0)]),
            critical_curves=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            caustics=ag.GridIrregularGrouped([(1.0, 1.0), (2.0, 2.0)]),
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
            plotter=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths

    def test__plot_voronoi_mapper__works_with_all_extras_included(
        self, voronoi_mapper_9_3x3, plot_path, plot_patch
    ):

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="mapper1", format="png")
        )

        plotter._plot_voronoi_mapper(
            mapper=voronoi_mapper_9_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper1.png") in plot_patch.paths

        plotter = aplt.Plotter2D(
            output=aplt.Output(path=plot_path, filename="mapper2", format="png")
        )

        plotter._plot_voronoi_mapper(
            mapper=voronoi_mapper_9_3x3,
            include_pixelization_grid=True,
            include_grid=True,
            include_border=True,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
        )

        assert path.join(plot_path, "mapper2.png") in plot_patch.paths

        aplt.MapperObj(
            mapper=voronoi_mapper_9_3x3,
            full_indexes=[[(0, 0), (0, 1)], [(1, 2)]],
            pixelization_indexes=[[0, 1], [2]],
            plotter=aplt.Plotter2D(
                output=aplt.Output(path=plot_path, filename="mapper3", format="png")
            ),
        )

        assert path.join(plot_path, "mapper3.png") in plot_patch.paths


class TestInclude:
    def test__critical_curves_from_object(self, lp_0, mp_0):

        include = aplt.Include2D(critical_curves=False)

        critical_curves = include.critical_curves_from_obj(obj=mp_0)

        assert critical_curves == None

        include = aplt.Include2D(critical_curves=True)

        critical_curves = include.critical_curves_from_obj(obj=lp_0)

        assert critical_curves == None

        include = aplt.Include2D(critical_curves=True)

        critical_curves = include.critical_curves_from_obj(obj=mp_0)

        assert critical_curves[0] == pytest.approx(mp_0.critical_curves[0], 1.0e-4)
        assert critical_curves[1] == pytest.approx(mp_0.critical_curves[1], 1.0e-4)

    def test__caustics_from_object(self, lp_0, mp_0):

        include = aplt.Include2D(caustics=False)

        caustics = include.caustics_from_obj(obj=mp_0)

        assert caustics == None

        include = aplt.Include2D(caustics=True)

        caustics = include.caustics_from_obj(obj=lp_0)

        assert caustics == None

        include = aplt.Include2D(caustics=True)

        caustics = include.caustics_from_obj(obj=mp_0)

        assert caustics[0] == pytest.approx(mp_0.caustics[0], 1.0e-4)
        assert caustics[1] == pytest.approx(mp_0.caustics[1], 1.0e-4)

    def test__new_include_with_preloaded_critical_curves_and_caustics(self):

        include = aplt.Include2D(mask=True)

        assert include.preloaded_critical_curves == None
        assert include.preloaded_caustics == None

        include = include.new_include_with_preloaded_critical_curves_and_caustics(
            preloaded_critical_curves=1, preloaded_caustics=2
        )

        assert include.mask == True
        assert include.preloaded_critical_curves == 1
        assert include.preloaded_caustics == 2
