import autoarray as aa
import autoarray.plot as aplt
from autoarray.plot import abstract_plotters

from autogalaxy.profiles.mass_profiles import MassProfile
from autogalaxy.plot.mat_wrap.lensing_visuals import Visuals1D
from autogalaxy.plot.mat_wrap.lensing_visuals import Visuals2D


class LensingObjPlotter(abstract_plotters.AbstractPlotter):

    lensing_obj = None
    grid = None

    @property
    def visuals_with_include_1d(self) -> Visuals1D:
        """
        Extract from the `LensingObj` attributes that can be plotted and return them in a `Visuals1D` object.

        Only attributes with `True` entries in the `Include` object are extracted for plotting.

        From a `LensingObjProfilePlotter` the following 1D attributes can be extracted for plotting:

        - einstein_radius: the Einstein radius of the `MassProfile`.

        Returns
        -------
        vis.Visuals1D
            The collection of attributes that can be plotted by a `Plotter1D` object.
        """
        if self.include_1d.einstein_radius:
            einstein_radius = self.lensing_obj.einstein_radius_from(grid=self.grid)
        else:
            einstein_radius = None

        return self.visuals_1d + self.visuals_1d.__class__(
            self.extract_1d("einstein_radius", value=einstein_radius)
        )

    @property
    def visuals_with_include_2d(self) -> Visuals2D:
        """
        Extract from the `LensingObj` attributes that can be plotted and return them in a `Visuals` object.

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

        return self.visuals_2d + self.visuals_2d.__class__(
            origin=self.extract_2d(
                "origin", value=aa.Grid2DIrregular(grid=[self.grid.origin])
            ),
            mask=self.extract_2d("mask", value=self.grid.mask),
            border=self.extract_2d(
                "border", value=self.grid.mask.border_grid_sub_1.binned
            ),
            mass_profile_centres=self.extract_2d(
                "mass_profile_centres",
                self.lensing_obj.extract_attribute(cls=MassProfile, attr_name="centre"),
            ),
            critical_curves=self.extract_2d(
                "critical_curves",
                self.lensing_obj.critical_curves_from(grid=self.grid),
                "critical_curves",
            ),
        )

    def figures_2d(
        self,
        convergence: bool = False,
        potential: bool = False,
        deflections_y: bool = False,
        deflections_x: bool = False,
        magnification: bool = False,
    ):

        if convergence:

            self.mat_plot_2d.plot_array(
                array=self.lensing_obj.convergence_2d_from(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(
                    title="Convergence", filename="convergence_2d"
                ),
            )

        if potential:

            self.mat_plot_2d.plot_array(
                array=self.lensing_obj.potential_2d_from(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(title="Potential", filename="potential_2d"),
            )

        if deflections_y:

            deflections = self.lensing_obj.deflections_2d_from(grid=self.grid)
            deflections_y = aa.Array2D.manual_mask(
                array=deflections.slim[:, 0], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_y,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(
                    title="Deflections Y", filename="deflections_y_2d"
                ),
            )

        if deflections_x:

            deflections = self.lensing_obj.deflections_2d_from(grid=self.grid)
            deflections_x = aa.Array2D.manual_mask(
                array=deflections.slim[:, 1], mask=self.grid.mask
            )

            self.mat_plot_2d.plot_array(
                array=deflections_x,
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(
                    title="deflections X", filename="deflections_x_2d"
                ),
            )

        if magnification:

            self.mat_plot_2d.plot_array(
                array=self.lensing_obj.magnification_2d_from(grid=self.grid),
                visuals_2d=self.visuals_with_include_2d,
                auto_labels=aplt.AutoLabels(
                    title="Magnification", filename="magnification_2d"
                ),
            )
