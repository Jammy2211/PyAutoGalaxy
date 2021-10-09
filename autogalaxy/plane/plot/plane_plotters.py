import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plane.plane import Plane
from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.plot.mat_wrap.lensing_mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.lensing_mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.lensing_visuals import Visuals1D
from autogalaxy.plot.mat_wrap.lensing_visuals import Visuals2D
from autogalaxy.plot.mat_wrap.lensing_include import Include1D
from autogalaxy.plot.mat_wrap.lensing_include import Include2D
from autogalaxy.plot.lensing_obj_plotter import LensingObjPlotter


class PlanePlotter(LensingObjPlotter):
    def __init__(
        self,
        plane: Plane,
        grid: aa.Grid2D,
        mat_plot_1d: MatPlot1D = MatPlot1D(),
        visuals_1d: Visuals1D = Visuals1D(),
        include_1d: Include1D = Include1D(),
        mat_plot_2d: MatPlot2D = MatPlot2D(),
        visuals_2d: Visuals2D = Visuals2D(),
        include_2d: Include2D = Include2D(),
    ):
        super().__init__(
            mat_plot_2d=mat_plot_2d,
            include_2d=include_2d,
            visuals_2d=visuals_2d,
            mat_plot_1d=mat_plot_1d,
            include_1d=include_1d,
            visuals_1d=visuals_1d,
        )

        self.plane = plane
        self.grid = grid

    @property
    def lensing_obj(self) -> Plane:
        return self.plane

    @property
    def visuals_with_include_2d(self) -> Visuals2D:
        """
        Extracts from a `Structure` attributes that can be plotted and return them in a `Visuals` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From an `AbstractStructure` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the structure's coordinate system.
        - mask: the mask of the structure.
        - border: the border of the structure's mask.

        Parameters
        ----------
        structure : abstract_structure.AbstractStructure
            The structure whose attributes are extracted for plotting.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter2D` object.
        """

        visuals_2d = super().visuals_with_include_2d

        visuals_2d.mask = None

        return visuals_2d + visuals_2d.__class__(
            grid=self.extract_2d("grid", self.grid),
            light_profile_centres=self.extract_2d(
                "light_profile_centres",
                self.plane.extract_attribute(cls=LightProfile, attr_name="centre"),
            ),
        )

    def figures_2d(
        self,
        image: bool = False,
        plane_image: bool = False,
        plane_grid: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        contribution_map: bool = False,
        title_suffix: str = "",
        filename_suffix: str = "",
    ):

        if image:

            self.mat_plot_2d.plot_array(
                array=self.plane.image_2d_from(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(
                    title=f"Image{title_suffix}", filename=f"image_2d{filename_suffix}"
                ),
            )

        if plane_image:

            self.mat_plot_2d.plot_array(
                array=self.plane.plane_image_2d_from(grid=self.grid).array,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(
                    title=f"Plane Image{title_suffix}",
                    filename=f"plane_image{filename_suffix}",
                ),
            )

        if plane_grid:

            self.mat_plot_2d.plot_grid(
                grid=self.grid,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(
                    title=f"Plane Grid2D{title_suffix}",
                    filename=f"plane_grid{filename_suffix}",
                ),
            )

        super().figures_2d(
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
        )

        if contribution_map:

            self.mat_plot_2d.plot_array(
                array=self.plane.contribution_map,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(
                    title="Contribution Map", filename="contribution_map_2d"
                ),
            )

    def subplot(
        self,
        image: bool = False,
        plane_image: bool = False,
        plane_grid: bool = False,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
        contribution_map: bool = False,
        auto_filename: str = "subplot_plane",
    ):

        self._subplot_custom_plot(
            image=image,
            plane_image=plane_image,
            plane_grid=plane_grid,
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
            contribution_map=contribution_map,
            auto_labels=aplt.AutoLabels(filename=auto_filename),
        )

    def subplot_with_source_grid(self):

        self.open_subplot_figure(number_subplots=2)

        self.figures_2d()
        self.mat_plot_2d.plot_grid(
            grid=self.plane.traced_grid_from(grid=self.grid),
            visuals_2d=self.visuals_with_include_2d,
            auto_labels=aplt.AutoLabels(),
        )

        self.mat_plot_2d.output.subplot_to_figure()
        self.close_subplot_figure()
