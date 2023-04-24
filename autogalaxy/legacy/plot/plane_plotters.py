import autoarray.plot as aplt

from autogalaxy.plane.plot.plane_plotters import PlanePlotter as PlanePlotterBase


class PlanePlotter(PlanePlotterBase):
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
            Whether to make a 2D plot (via `imshow`) of the image of plane in its image-plane (e.g. after
            lensing).
        plane_image
            Whether to make a 2D plot (via `imshow`) of the image of the plane in the soure-plane (e.g. its
            unlensed light).
        convergence
            Whether to make a 2D plot (via `imshow`) of the convergence.
        potential
            Whether to make a 2D plot (via `imshow`) of the potential.
        deflections_y
            Whether to make a 2D plot (via `imshow`) of the y component of the deflection angles.
        deflections_x
            Whether to make a 2D plot (via `imshow`) of the x component of the deflection angles.
        magnification
            Whether to make a 2D plot (via `imshow`) of the magnification.
        contribution_map
            Whether to make a 2D plot (via `imshow`) of the contribution map.
        title_suffix
            Add a suffix to the end of the matplotlib title label.
        filename_suffix
            Add a suffix to the end of the filename the plot is saved to hard-disk using.
        """

        super().figures_2d(
            image=image,
            plane_image=plane_image,
            plane_grid=plane_grid,
            convergence=convergence,
            potential=potential,
            deflections_y=deflections_y,
            deflections_x=deflections_x,
            magnification=magnification,
            title_suffix=title_suffix,
            filename_suffix=filename_suffix,
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
