from typing import List, Union

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.plot.visuals.two_d import Visuals2D
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.galaxy.galaxy import Galaxy


class GetVisuals2D(aplt.GetVisuals2D):
    def __init__(self, visuals: Visuals2D):
        """
        Class which gets 2D attributes and adds them to a `Visuals2D` objects, such that they are plotted on 2D figures.

        The `GetVisuals2D` class adds new visuals to a pre-existing `Visuals2D` object that is passed to
        its `__init__` method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        include
            Sets which 2D visuals are included on the figure that is to be plotted (only entries which are `True`
            are extracted via the `GetVisuals2D` object).
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals2D` class.
        """
        super().__init__(visuals=visuals)

    def via_light_obj_from(
        self, light_obj: Union[LightProfile, Galaxy], grid
    ) -> Visuals2D:
        """
        From an object with light profiles (e.g. a `LightProfile`, `Galaxy`) get its attributes that can be
        plotted and return them  in a `Visuals2D` object.

        Only attributes not already in `self.visuals` are extracted for plotting.

        From a light object the following 2D attributes can be extracted for plotting:

        - origin: the (y,x) origin of the coordinate system used to plot the light object's quantities in 2D.
        - mask: the mask of the grid used to plot the light object's quantities in 2D.
        - border: the border of this mask.
        - light profile centres: the (y,x) centre of every `LightProfile` in the object.

        Parameters
        ----------
        light_obj
            The light object (e.g. a `LightProfile`, `Galaxy`) whose attributes are extracted for plotting.
        grid
            The 2D grid of (y,x) coordinates used to plot the light object's quantities in 2D.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter` object.
        """

        visuals_via_mask = self.via_mask_from(mask=grid.mask)

        if isinstance(light_obj, LightProfile):
            light_profile_centres = self.get(
                "light_profile_centres", aa.Grid2DIrregular(values=[light_obj.centre])
            )

        else:
            light_profile_centres = self.get(
                "light_profile_centres",
                light_obj.extract_attribute(cls=LightProfile, attr_name="centre"),
            )

        return (
            self.visuals
            + visuals_via_mask
            + self.visuals.__class__(light_profile_centres=light_profile_centres)
        )

    def via_mass_obj_from(
        self, mass_obj: Union[MassProfile, Galaxy], grid: aa.type.Grid2DLike
    ) -> Visuals2D:
        """
        From an object with mass profiles (e.g. a `MassProfile`, `Galaxy`) get its attributes that can be
        plotted and return them  in a `Visuals2D` object.

        Only attributes not already in `self.visuals` are extracted for plotting.

        From a mass object the following 2D attributes can be extracted for plotting:

        - origin: the (y,x) origin of the coordinate system used to plot the mass object's quantities in 2D.
        - mask: the mask of the grid used to plot the mass object's quantities in 2D.
        - border: the border of this mask.
        - mass profile centres: the (y,x) centre of every `MassProfile` in the mass object.
        - critical curves: the critical curves of the mass object.

        Parameters
        ----------
        mass_obj
            The mass object (e.g. a `MassProfile`, `Galaxy`) whose attributes are extracted for plotting.
        grid
            The 2D grid of (y,x) coordinates used to plot the mass object's quantities in 2D.

        Returns
        -------
        vis.Visuals2D
            The collection of attributes that can be plotted by a `Plotter` object.
        """

        visuals_via_mask = self.via_mask_from(mask=grid.mask)

        if isinstance(mass_obj, MassProfile):
            mass_profile_centres = self.get(
                "mass_profile_centres", aa.Grid2DIrregular(values=[mass_obj.centre])
            )

        else:
            mass_profile_centres = self.get(
                "mass_profile_centres",
                mass_obj.extract_attribute(cls=MassProfile, attr_name="centre"),
            )

        tangential_critical_curves = self.get(
            "tangential_critical_curves",
            mass_obj.tangential_critical_curve_list_from(grid=grid),
        )

        radial_critical_curves = None

        radial_critical_curve_area_list = mass_obj.radial_critical_curve_area_list_from(
            grid=grid
        )

        if any([area > grid.pixel_scale for area in radial_critical_curve_area_list]):
            radial_critical_curves = self.get(
                "radial_critical_curves",
                mass_obj.radial_critical_curve_list_from(grid=grid),
            )

        return (
            self.visuals
            + visuals_via_mask
            + self.visuals.__class__(
                mass_profile_centres=mass_profile_centres,
                tangential_critical_curves=tangential_critical_curves,
                radial_critical_curves=radial_critical_curves,
            )
        )

    def via_light_mass_obj_from(self, light_mass_obj: Union[Galaxy], grid) -> Visuals2D:
        """
        From an object that contains both light profiles and / or mass profiles (e.g. a `Galaxy`), get the
        attributes that can be plotted and returns them in a `Visuals2D` object.

        Only attributes with `True` entries in the `Include` object are extracted.

        From a light and lensing object the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the coordinate system used to plot the light object's quantities in 2D.
        - light profile centres: the (y,x) centre of every `LightProfile` in the object.
        - mass profile centres: the (y,x) centre of every `MassProfile` in the object.
        - critical curves: the critical curves of all mass profile combined.

        Parameters
        ----------
        light_mass_obj
            The light and mass object (e.g. a `Galaxy`) whose attributes are extracted for plotting.
        grid
            The 2D grid of (y,x) coordinates used to plot the light and mass object's quantities in 2D.

        Returns
        -------
        vis.Visuals2D
            A collection of attributes that can be plotted by a `Plotter` object.
        """

        visuals_2d = self.via_mass_obj_from(mass_obj=light_mass_obj, grid=grid)
        visuals_2d.mask = None

        visuals_with_grid = self.visuals.__class__(grid=self.get("grid", grid))

        return (
            visuals_2d
            + visuals_with_grid
            + self.via_light_obj_from(light_obj=light_mass_obj, grid=grid)
        )

    def via_galaxies_from(
        self, galaxies: List[Galaxy], grid: aa.type.Grid2DLike, galaxy_index: int
    ) -> Visuals2D:
        """
        From a list of galaxies get the attributes that can be plotted and returns them in a `Visuals2D` object.

        Only attributes with `True` entries in the `Include` object are extracted.

        From a list of galaxie the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the coordinate system used to plot the light object's quantities in 2D.
        - border: the border of the mask of the grid used to plot the light object's quantities in 2D.
        - light profile centres: the (y,x) centre of every `LightProfile` in the object.
        - mass profile centres: the (y,x) centre of every `MassProfile` in the object.
        - critical curves: the critical curves of all of the galaxy's mass profiles combined.
        - caustics: the caustics of all of the galaxy's mass profiles combined.

        When plotting galaxies it is common for plots to only display quantities corresponding to one galaxy at a time
        (e.g. the image of each galaxy). Therefore, quantities are only extracted from one galaxy, specified by the
        input `galaxy_index`.

        Parameters
        ----------
        galaxies
            The galaxies which have attributes extracted for plotting.
        grid
            The 2D grid of (y,x) coordinates used to plot the galaxies quantities in 2D.
        galaxy_index
            The index of the galaxy in the galaxies which is used to extract quantities, as only one galaxy is plotted
            at a time.

        Returns
        -------
        vis.Visuals2D
            A collection of attributes that can be plotted by a `Plotter` object.
        """
        origin = self.get("origin", value=aa.Grid2DIrregular(values=[grid.origin]))

        light_profile_centres = self.get(
            "light_profile_centres",
            galaxies[galaxy_index].extract_attribute(
                cls=LightProfile, attr_name="centre"
            ),
        )

        mass_profile_centres = self.get(
            "mass_profile_centres",
            galaxies[galaxy_index].extract_attribute(
                cls=MassProfile, attr_name="centre"
            ),
        )

        return self.visuals + self.visuals.__class__(
            origin=origin,
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
        )

    def via_fit_imaging_from(self, fit: FitImaging) -> Visuals2D:
        """
        From a `FitImaging` get its attributes that can be plotted and return them in a `Visuals2D` object.

        From a `FitImaging` the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the 2D coordinate system.
        - mask: the 2D mask.
        - border: the border of the 2D mask, which are all of the mask's exterior edge pixels.
        - light profile centres: the (y,x) centre of every `LightProfile` in the object.
        - mass profile centres: the (y,x) centre of every `MassProfile` in the object.
        - critical curves: the critical curves of all mass profile combined.

        Parameters
        ----------
        fit
            The fit imaging object whose attributes are extracted for plotting.

        Returns
        -------
        Visuals2D
            The collection of attributes that are plotted by a `Plotter` object.
        """
        visuals_2d_via_fit = super().via_fit_imaging_from(fit=fit)

        visuals_2d_via_light_mass_obj = self.via_light_mass_obj_from(
            light_mass_obj=fit.galaxies, grid=fit.grids.lp
        )

        return visuals_2d_via_fit + visuals_2d_via_light_mass_obj
