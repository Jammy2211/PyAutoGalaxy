import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.geometry_profiles import EllProfile
from autogalaxy.operate.deflections import OperateDeflections

from autogalaxy import exc


class MassProfile(EllProfile, OperateDeflections):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

    def deflections_yx_2d_from(self, grid):
        raise NotImplementedError

    def deflections_2d_via_potential_2d_from(self, grid):

        potential = self.potential_2d_from(grid=grid)

        deflections_y_2d = np.gradient(potential.native, grid.native[:, 0, 0], axis=0)
        deflections_x_2d = np.gradient(potential.native, grid.native[0, :, 1], axis=1)

        return aa.Grid2D.manual_mask(
            grid=np.stack((deflections_y_2d, deflections_x_2d), axis=-1), mask=grid.mask
        )

    def convergence_2d_from(self, grid):
        raise NotImplementedError

    def convergence_func(self, grid_radius: float) -> float:
        raise NotImplementedError

    @aa.grid_dec.grid_1d_to_structure
    def convergence_1d_from(self, grid: aa.type.Grid1D2DLike) -> aa.type.Grid1D2DLike:
        return self.convergence_2d_from(grid=grid)

    def potential_2d_from(self, grid):
        raise NotImplementedError

    @aa.grid_dec.grid_1d_to_structure
    def potential_1d_from(self, grid: aa.type.Grid1D2DLike) -> aa.type.Grid1D2DLike:
        return self.potential_2d_from(grid=grid)

    def potential_func(self, u, y, x):
        raise NotImplementedError

    def mass_integral(self, x):
        return 2 * np.pi * x * self.convergence_func(grid_radius=x)

    @property
    def ellipticity_rescale(self):
        return NotImplementedError()

    def mass_angular_within_circle_from(self, radius: float):
        """
        Integrate the mass profiles's convergence profile to compute the total mass within a circle of
        specified radius. This is centred on the mass profile.

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        """

        return quad(self.mass_integral, a=0.0, b=radius)[0]

    def density_between_circular_annuli(
        self, inner_annuli_radius: float, outer_annuli_radius: float
    ):
        """Calculate the mass between two circular annuli and compute the density by dividing by the annuli surface
        area.

        The value returned by the mass integral is dimensionless, therefore the density between annuli is returned in \
        unit_label of inverse radius squared. A conversion factor can be specified to convert this to a physical value \
        (e.g. the critical surface mass density).

        Parameters
        -----------
        inner_annuli_radius
            The radius of the inner annulus outside of which the density are estimated.
        outer_annuli_radius
            The radius of the outer annulus inside of which the density is estimated.
        """
        annuli_area = (np.pi * outer_annuli_radius**2.0) - (
            np.pi * inner_annuli_radius**2.0
        )

        outer_mass = self.mass_angular_within_circle_from(radius=outer_annuli_radius)

        inner_mass = self.mass_angular_within_circle_from(radius=inner_annuli_radius)

        return (outer_mass - inner_mass) / annuli_area

    @property
    def average_convergence_of_1_radius(self):
        """
        The radius a critical curve forms for this mass profile, e.g. where the mean convergence is equal to 1.0.

        In case of ellipitical mass profiles, the 'average' critical curve is used, whereby the convergence is \
        rescaled into a circle using the axis ratio.

        This radius corresponds to the Einstein radius of the mass profile, and is a property of a number of \
        mass profiles below.
        """

        def func(radius):

            return (
                self.mass_angular_within_circle_from(radius=radius)
                - np.pi * radius**2.0
            )

        return self.ellipticity_rescale * root_scalar(func, bracket=[1e-4, 1e4]).root

    def mass_angular_via_normalization_from(self, normalization, radius):

        mass_profile = self.with_new_normalization(normalization=normalization)

        return mass_profile.mass_angular_within_circle_from(radius=radius)

    def normalization_via_mass_angular_from(
        self,
        mass_angular,
        radius,
        normalization_min=1e-15,
        normalization_max=1e15,
        bins=200,
    ):

        normalization_list = np.logspace(
            np.log10(normalization_min), np.log10(normalization_max), bins
        )

        mass_angulars = [
            self.mass_angular_via_normalization_from(
                normalization=normalization, radius=radius
            )
            for normalization in normalization_list
        ]

        normalization_list = [
            normalization
            for normalization, mass in zip(normalization_list, mass_angulars)
            if mass is not None
        ]
        mass_angulars = list(filter(None, mass_angulars))

        if (
            (len(mass_angulars) < 2)
            or (mass_angulars[0] > mass_angular)
            or (mass_angulars[-1] < mass_angular)
        ):
            raise exc.ProfileException(
                "The normalization could not be computed from the Einstein Radius via the average of the convergence. "
                ""
                "The input einstein_radius may be too small or large to feasibly be computed by integrating the "
                "convergence. Alternative the normalization range or number of bins may need to be changed to "
                "capture the true einstein_radius value."
            )

        def func(normalization, mass_angular_root, radius):

            mass_angular = self.mass_angular_via_normalization_from(
                normalization=normalization, radius=radius
            )

            return mass_angular - mass_angular_root

        return root_scalar(
            func,
            bracket=[normalization_list[0], normalization_list[-1]],
            args=(mass_angular, radius),
        ).root

    def with_new_normalization(self, normalization):
        raise NotImplementedError()

    def einstein_radius_via_normalization_from(self, normalization):

        mass_profile = self.with_new_normalization(normalization=normalization)

        try:
            return mass_profile.average_convergence_of_1_radius
        except ValueError:
            return None

    def normalization_via_einstein_radius_from(
        self, einstein_radius, normalization_min=1e-9, normalization_max=1e9, bins=100
    ):

        normalization_list = np.logspace(
            np.log10(normalization_min), np.log10(normalization_max), bins
        )

        einstein_radii = [
            self.einstein_radius_via_normalization_from(normalization=normalization)
            for normalization in normalization_list
        ]

        normalization_list = [
            normalization
            for normalization, radii in zip(normalization_list, einstein_radii)
            if radii is not None
        ]
        einstein_radii = list(filter(None, einstein_radii))

        if (
            (len(einstein_radii) < 2)
            or (einstein_radii[0] > einstein_radius)
            or (einstein_radii[-1] < einstein_radius)
        ):
            raise exc.ProfileException(
                "The normalization could not be computed from the Einstein Radius via the average of the convergence. "
                ""
                "The input einstein_radius may be too small or large to feasibly be computed by integrating the "
                "convergence. Alternative the normalization range or number of bins may need to be changed to "
                "capture the true einstein_radius value."
            )

        def func(normalization, einstein_radius_root):

            einstein_radius = self.einstein_radius_via_normalization_from(
                normalization=normalization
            )

            return einstein_radius - einstein_radius_root

        return root_scalar(
            func,
            bracket=[normalization_list[0], normalization_list[-1]],
            args=(einstein_radius,),
        ).root

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class and its children profiles in the the galaxy as a `ValueIrregular`
        or `Grid2DIrregular` object.

        For example, if a galaxy has two light profiles and we want the `LightProfile` axis-ratios, the following:

        `galaxy.extract_attribute(cls=LightProfile, name="axis_ratio"`

        would return:

        ValuesIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want the `MassProfile` centres, the following:

        `galaxy.extract_attribute(cls=MassProfile, name="centres"`

         would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all light profiles colored by their profile.
        """

        if isinstance(self, cls):
            if hasattr(self, attr_name):

                attribute = getattr(self, attr_name)

                if isinstance(attribute, float):
                    return aa.ValuesIrregular(values=[attribute])
                if isinstance(attribute, tuple):
                    return aa.Grid2DIrregular(grid=[attribute])
