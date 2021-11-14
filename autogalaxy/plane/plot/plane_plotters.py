import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.plot.abstract_plotters import Plotter
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D
from autogalaxy.plot.mat_wrap.visuals import Visuals1D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D
from autogalaxy.plot.mat_wrap.include import Include1D
from autogalaxy.plot.mat_wrap.include import Include2D
from autogalaxy.plot.mass_plotter import MassPlotter

from autogalaxy.plane.plane import Plane


class PlanePlotter(Plotter):
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
        """
        Plots the attributes of `Plane` objects using the matplotlib methods `plot()` and `imshow()` and many 
        other matplotlib functions which customize the plot's appearance.

        The `mat_plot_1d` and `mat_plot_2d` attributes wrap matplotlib function calls to make the figure. By default, 
        the settings passed to every matplotlib function called are those specified in 
        the `config/visualize/mat_wrap/*.ini` files, but a user can manually input values into `MatPlot2D` to 
        customize the figure's appearance.

        Overlaid on the figure are visuals, contained in the `Visuals1D` and `Visuals2D` objects. Attributes may be 
        extracted from the `MassProfile` and plotted via the visuals object, if the corresponding entry is `True` in 
        the `Include1D` or `Include2D` object or the `config/visualize/include.ini` file.

        Parameters
        ----------
        plane
            The plane the plotter plots.
        grid
            The 2D (y,x) grid of coordinates used to evaluate the plane's light and mass quantities that are plotted.
        mat_plot_1d
            Contains objects which wrap the matplotlib function calls that make 1D plots.
        visuals_1d
            Contains 1D visuals that can be overlaid on 1D plots.
        include_1d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 1D plots.
        mat_plot_2d
            Contains objects which wrap the matplotlib function calls that make 2D plots.
        visuals_2d
            Contains 2D visuals that can be overlaid on 2D plots.
        include_2d
            Specifies which attributes of the `MassProfile` are extracted and plotted as visuals for 2D plots.
        """
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

        self._mass_plotter = MassPlotter(
            mass_obj=self.plane,
            grid=self.grid,
            get_visuals_2d=self.get_visuals_2d,
            mat_plot_2d=self.mat_plot_2d,
            include_2d=self.include_2d,
            visuals_2d=self.visuals_2d,
        )

    def get_visuals_2d(self) -> Visuals2D:
        return self.get_2d.via_light_mass_obj_from(
            light_mass_obj=self.plane, grid=self.grid
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
        """
        Plots the individual attributes of the plotter's `Plane` object in 2D, which are computed via the plotter's 2D
        grid object.

        The API is such that every plottable attribute of the `Plane` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is plotted.

        Parameters
        ----------
        image
            Whether or not to make a 2D plot (via `imshow`) of the image of plane in its image-plane (e.g. after
            lensing).
        plane_image
            Whether or not to make a 2D plot (via `imshow`) of the image of the plane in the soure-plane (e.g. its
            unlensed light).
        convergence
            Whether or not to make a 2D plot (via `imshow`) of the convergence.
        potential
            Whether or not to make a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether or not to make a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether or not to make a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether or not to make a 2D plot (via `imshow`) of the magnification.
        contribution_map
            Whether or not to make a 2D plot (via `imshow`) of the contribution map.
        title_suffix
            Add a suffix to the end of the matplotlib title label.
        filename_suffix
            Add a suffix to the end of the filename the plot is saved to hard-disk using.
        """
        if image:

            self.mat_plot_2d.plot_array(
                array=self.plane.image_2d_from(grid=self.grid),
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Image{title_suffix}", filename=f"image_2d{filename_suffix}"
                ),
            )

        if plane_image:

            self.mat_plot_2d.plot_array(
                array=self.plane.plane_image_2d_from(grid=self.grid).array,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Plane Image{title_suffix}",
                    filename=f"plane_image{filename_suffix}",
                ),
            )

        if plane_grid:

            self.mat_plot_2d.plot_grid(
                grid=self.grid,
                visuals_2d=self.get_visuals_2d(),
                auto_labels=aplt.AutoLabels(
                    title=f"Plane Grid2D{title_suffix}",
                    filename=f"plane_grid{filename_suffix}",
                ),
            )

        self._mass_plotter.figures_2d(
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
        )

        if contribution_map:

            self.mat_plot_2d.plot_array(
                array=self.plane.contribution_map,
                visuals_2d=self.get_visuals_2d(),
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
        """
        Plots the individual attributes of the plotter's `Plane` object in 2D on a subplot, which are computed via the 
        plotter's 2D grid object.

        The API is such that every plottable attribute of the `Plane` object is an input parameter of type bool of
        the function, which if switched to `True` means that it is included on the subplot.

        Parameters
        ----------
        image
            Whether or not to  include a 2D plot (via `imshow`) of the image of plane in its image-plane (e.g. after
            lensing).
        plane_image
            Whether or not to  include a 2D plot (via `imshow`) of the image of the plane in the soure-plane (e.g. its
            unlensed light).
        convergence
            Whether or not to  include a 2D plot (via `imshow`) of the convergence.
        potential
            Whether or not to  include a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether or not to  include a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether or not to  include a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether or not to  include a 2D plot (via `imshow`) of the magnification.
        contribution_map
            Whether or not to  include a 2D plot (via `imshow`) of the contribution map.
        auto_filename
            The default filename of the output subplot if written to hard-disk.
        """
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
