from typing import List, Union

import autoarray as aa

from autoarray.plot.mat_wrap import get_visuals as gv

from autogalaxy.plot.mat_wrap.include import Include1D
from autogalaxy.plot.mat_wrap.include import Include2D
from autogalaxy.plot.mat_wrap.visuals import Visuals1D
from autogalaxy.plot.mat_wrap.visuals import Visuals2D

from autogalaxy.util import error_util

from autoarray.structures.grids.irregular_2d import Grid2DIrregular

from autogalaxy.imaging.fit_imaging import FitImaging
from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.profiles.mass_profiles.mass_profiles import MassProfile
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plane.plane import Plane


class GetVisuals1D(gv.GetVisuals1D):
    def __init__(self, include: Include1D, visuals: Visuals1D):
        """
        Class which gets 1D attributes and adds them to a `Visuals1D` objects, such that they are plotted on 1D figures.

        For a visual to be extracted and added for plotting, it must have a `True` value in its corresponding entry in
        the `Include1D` object. If this entry is `False`, the `GetVisuals1D.get` method returns a None and the attribute
        is omitted from the plot.

        The `GetVisuals1D` class adds new visuals to a pre-existing `Visuals1D` object that is passed to its `__init__`
        method. This only adds a new entry if the visual are not already in this object.

        Parameters
        ----------
        include
            Sets which 1D visuals are included on the figure that is to be plotted (only entries which are `True`
            are extracted via the `GetVisuals1D` object).
        visuals
            The pre-existing visuals of the plotter which new visuals are added too via the `GetVisuals1D` class.
        """
        super().__init__(include=include, visuals=visuals)

    def via_light_obj_from(self, light_obj: Union[LightProfile, Galaxy]) -> Visuals1D:
        """
        From an object with light profiles (e.g. a `LightProfile`, `Galaxy`) get its attributes that can be plotted 
        and return them  in a `Visuals1D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.

        From a light object the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the light objects total integrated luminosity.

        Parameters
        ----------
        light_obj
            The light object (e.g. a `LightProfile`, `Galaxy`) whose attributes are extracted for plotting.

        Returns
        -------
        Visuals1D
            The collection of attributes that can be plotted by a `Plotter` object.
        """

        half_light_radius = self.get(
            "half_light_radius", value=light_obj.half_light_radius
        )

        return self.visuals + self.visuals.__class__(
            half_light_radius=half_light_radius
        )

    def via_light_obj_list_from(
        self, light_obj_list: Union[List[LightProfile], List[Galaxy]], low_limit: float
    ) -> Visuals1D:
        """
        From a list of objects with light profiles (e.g. a `LightProfile`, `Galaxy`) get its attributes that can be 
        plotted and return them  in a `Visuals1D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.
        
        This function iterates over all light objects in the list and averages over each attribute's values to estimate 
        the mean value of the attribute and its error, both of which can then be plotted. This is typically used
        to plot 1D errors on a quantity that are estimated via a Probability Density Function.

        From a light object lust the following 1D attributes can be extracted for plotting:

        - half_light_radius: the radius containing 50% of the light objects total integrated luminosity.

        Parameters
        ----------
        light_obj_list
            The list of light objects (e.g. a `LightProfile`, `Galaxy`) whose mean attributes and error estimates are
            extracted for plotting.
        low_limit
            The value of sigma to which errors are estimated (e.g. 1.0 will estimate errors at the ~0.32 and ~0.68
            intervals of the probability distribution.

        Returns
        -------
        Visuals1D
            The mean value and errors of each attribute that are plotted in 1D by a `Plotter` object.
        """

        if self.include.half_light_radius:

            half_light_radius_list = [
                light_profile.half_light_radius for light_profile in light_obj_list
            ]

            if None in half_light_radius_list:

                half_light_radius = None
                half_light_radius_errors = None

            else:

                half_light_radius, half_light_radius_errors = error_util.value_median_and_error_region_via_quantile(
                    value_list=half_light_radius_list, low_limit=low_limit
                )

        else:

            half_light_radius = None
            half_light_radius_errors = None

        half_light_radius = self.get("half_light_radius", value=half_light_radius)
        half_light_radius_errors = self.get(
            "half_light_radius", value=half_light_radius_errors
        )

        return self.visuals + self.visuals.__class__(
            half_light_radius=half_light_radius,
            half_light_radius_errors=half_light_radius_errors,
        )

    def via_mass_obj_from(
        self, mass_obj: Union[MassProfile, Galaxy], grid: aa.type.Grid2DLike
    ) -> Visuals1D:
        """
        From an object with mass profiles (e.g. a `MassProfile`, `Galaxy`) get its attributes that can be plotted 
        and return them  in a `Visuals1D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.

        From a mass object the following 1D attributes can be extracted for plotting:

        - einstein_radius: the einstein radius (e.g. area within critical curve) of the mass object.

        Mass profiles can be too shallow to do lensing and therefore an Einstein radius cannot be computed. This
        raises a TypeError which is accounted for below.

        Parameters
        ----------
        mass_obj
            The mass object (e.g. a `MassProfile`, `Galaxy`) whose attributes are extracted for plotting.

        Returns
        -------
        Visuals1D
            The collection of attributes that can be plotted by a `Plotter` object.
        """

        if self.include.einstein_radius:
            try:
                einstein_radius = mass_obj.einstein_radius_from(grid=grid)
            except (TypeError, AttributeError):
                einstein_radius = None
        else:
            einstein_radius = None

        einstein_radius = self.get("einstein_radius", value=einstein_radius)

        return self.visuals + self.visuals.__class__(einstein_radius=einstein_radius)

    def via_mass_obj_list_from(
        self,
        mass_obj_list: Union[List[MassProfile], List[Galaxy]],
        grid: aa.type.Grid2DLike,
        low_limit: float,
    ) -> Visuals1D:
        """
        From a list of objects with mass profiles (e.g. a `MassProfile`, `Galaxy`) get its attributes that can be 
        plotted and return them  in a `Visuals1D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.

        This function iterates over all mass objects in the list and averages over each attribute's values to estimate 
        the mean value of the attribute and its error, both of which can then be plotted. This is typically used
        to plot 1D errors on a quantity that are estimated via a Probability Density Function.

        From a mass object lust the following 1D attributes can be extracted for plotting:

        - half_mass_radius: the radius containing 50% of the mass objects total integrated luminosity.

        Parameters
        ----------
        mass_obj_list
            The list of mass objects (e.g. a `MassProfile`, `Galaxy`) whose mean attributes and error estimates are
            extracted for plotting.
        low_limit
            The value of sigma to which errors are estimated (e.g. 1.0 will estimate errors at the ~0.32 and ~0.68
            intervals of the probability distribution.

        Returns
        -------
        Visuals1D
            The mean value and errors of each attribute that are plotted in 1D by a `Plotter` object.
        """

        if self.include.einstein_radius:

            einstein_radius_list = []

            for mass_obj in mass_obj_list:

                try:
                    einstein_radius_list.append(
                        mass_obj.einstein_radius_from(grid=grid)
                    )
                except TypeError:
                    einstein_radius_list.append(None)

            einstein_radius_list = list(filter(None, einstein_radius_list))

            einstein_radius, einstein_radius_errors = error_util.value_median_and_error_region_via_quantile(
                value_list=einstein_radius_list, low_limit=low_limit
            )

        else:

            einstein_radius = None
            einstein_radius_errors = None

        einstein_radius = self.get("einstein_radius", value=einstein_radius)
        einstein_radius_errors = self.get(
            "einstein_radius", value=einstein_radius_errors
        )

        return self.visuals + self.visuals.__class__(
            einstein_radius=einstein_radius,
            einstein_radius_errors=einstein_radius_errors,
        )


class GetVisuals2D(gv.GetVisuals2D):
    def __init__(self, include: Include2D, visuals: Visuals2D):
        """
        Class which gets 2D attributes and adds them to a `Visuals2D` objects, such that they are plotted on 2D figures.

        For a visual to be extracted and added for plotting, it must have a `True` value in its corresponding entry in
        the `Include2D` object. If this entry is `False`, the `GetVisuals2D.get` method returns a None and the
        attribute is omitted from the plot.

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
        super().__init__(include=include, visuals=visuals)

    def via_light_obj_from(
        self, light_obj: Union[LightProfile, Galaxy, Plane], grid
    ) -> Visuals2D:
        """
        From an object with light profiles (e.g. a `LightProfile`, `Galaxy`, `Plane`) get its attributes that can be 
        plotted and return them  in a `Visuals2D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.

        From a light object the following 2D attributes can be extracted for plotting:

        - origin: the (y,x) origin of the coordinate system used to plot the light object's quantities in 2D.
        - mask: the mask of the grid used to plot the light object's quantities in 2D.
        - border: the border of this mask.
        - light profile centres: the (y,x) centre of every `LightProfile` in the object.

        Parameters
        ----------
        light_obj
            The light object (e.g. a `LightProfile`, `Galaxy`, `Plane`) whose attributes are extracted for plotting.
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
                "light_profile_centres", Grid2DIrregular(grid=[light_obj.centre])
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
        self, mass_obj: Union[MassProfile, Galaxy, Plane], grid: aa.type.Grid2DLike
    ) -> Visuals2D:
        """
        From an object with mass profiles (e.g. a `MassProfile`, `Galaxy`, `Plane`) get its attributes that can be 
        plotted and return them  in a `Visuals2D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include1D` object are extracted
        for plotting.

        From a mass object the following 2D attributes can be extracted for plotting:

        - origin: the (y,x) origin of the coordinate system used to plot the mass object's quantities in 2D.
        - mask: the mask of the grid used to plot the mass object's quantities in 2D.
        - border: the border of this mask.
        - mass profile centres: the (y,x) centre of every `MassProfile` in the mass object.
        - critical curves: the critical curves of the mass object.

        Parameters
        ----------
        mass_obj
            The mass object (e.g. a `MassProfile`, `Galaxy`, `Plane`) whose attributes are extracted for plotting.
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
                "mass_profile_centres", Grid2DIrregular(grid=[mass_obj.centre])
            )

        else:

            mass_profile_centres = self.get(
                "mass_profile_centres",
                mass_obj.extract_attribute(cls=MassProfile, attr_name="centre"),
            )

        critical_curves = self.get(
            "critical_curves",
            mass_obj.critical_curves_from(grid=grid),
            "critical_curves",
        )

        return (
            self.visuals
            + visuals_via_mask
            + self.visuals.__class__(
                mass_profile_centres=mass_profile_centres,
                critical_curves=critical_curves,
            )
        )

    def via_light_mass_obj_from(
        self, light_mass_obj: Union[Galaxy, Plane], grid
    ) -> Visuals2D:
        """
        From an object that contains both light profiles and / or mass profiles (e.g. a `Galaxy`, `Plane`), get the
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
            The light and mass object (e.g. a `Galaxy`, `Plane`) whose attributes are extracted for plotting.
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

    def via_plane_from(
        self, plane: Plane, grid: aa.type.Grid2DLike, galaxy_index: int
    ) -> Visuals2D:
        """
        From a `Plane` get the attributes that can be plotted and returns them in a `Visuals2D` object.

        Only attributes with `True` entries in the `Include` object are extracted.

        From a plane the following attributes can be extracted for plotting:

        - origin: the (y,x) origin of the coordinate system used to plot the light object's quantities in 2D.
        - border: the border of the mask of the grid used to plot the light object's quantities in 2D.
        - light profile centres: the (y,x) centre of every `LightProfile` in the object.
        - mass profile centres: the (y,x) centre of every `MassProfile` in the object.
        - critical curves: the critical curves of all of the plane's mass profiles combined.
        - caustics: the caustics of all of the plane's mass profiles combined.

        When plotting a `Plane` it is common for plots to only display quantities corresponding to one galaxy at a time
        (e.g. the image of each galaxy). Therefore, quantities are only extracted from one plane, specified by the
        input `galaxy_index`.

        Parameters
        ----------
        plane
            The `Plane` object which has attributes extracted for plotting.
        grid
            The 2D grid of (y,x) coordinates used to plot the plane's quantities in 2D.
        galaxy_index
            The index of the plane in the plane which is used to extract quantities, as only one plane is plotted
            at a time.

        Returns
        -------
        vis.Visuals2D
            A collection of attributes that can be plotted by a `Plotter` object.
        """
        origin = self.get("origin", value=aa.Grid2DIrregular(grid=[grid.origin]))

        light_profile_centres = self.get(
            "light_profile_centres",
            plane.galaxies[galaxy_index].extract_attribute(
                cls=LightProfile, attr_name="centre"
            ),
        )

        mass_profile_centres = self.get(
            "mass_profile_centres",
            plane.galaxies[galaxy_index].extract_attribute(
                cls=MassProfile, attr_name="centre"
            ),
        )

        if galaxy_index == 0:

            critical_curves = self.get(
                "critical_curves",
                plane.critical_curves_from(grid=grid),
                "critical_curves",
            )
        else:
            critical_curves = None

        if galaxy_index == 1:
            caustics = self.get("caustics", plane.caustics_from(grid=grid), "caustics")
        else:
            caustics = None

        return self.visuals + self.visuals.__class__(
            origin=origin,
            light_profile_centres=light_profile_centres,
            mass_profile_centres=mass_profile_centres,
            critical_curves=critical_curves,
            caustics=caustics,
        )

    def via_fit_imaging_from(self, fit: FitImaging) -> Visuals2D:
        """
        From a `FitImaging` get its attributes that can be plotted and return them in a `Visuals2D` object.

        Only attributes not already in `self.visuals` and with `True` entries in the `Include2D` object are extracted
        for plotting.

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
            light_mass_obj=fit.plane, grid=fit.grid
        )

        return visuals_2d_via_fit + visuals_2d_via_light_mass_obj
