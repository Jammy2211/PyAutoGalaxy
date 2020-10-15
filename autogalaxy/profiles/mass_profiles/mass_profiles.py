import typing

import numpy as np
from astropy import cosmology as cosmo
from scipy.integrate import quad
from scipy.optimize import root_scalar

import autofit as af
from autoarray.structures import grids
from autogalaxy import dimensions as dim
from autogalaxy import lensing
from autogalaxy.profiles import geometry_profiles
from autogalaxy.util import cosmology_util


class MassProfile(lensing.LensingObject):
    @property
    def mass_profiles(self):
        return [self]

    @property
    def has_mass_profile(self):
        return True

    @property
    def is_point_mass(self):
        return False

    @property
    def ellipticity_rescale(self):
        return NotImplementedError()

    @property
    def is_mass_sheet(self):
        return False


# noinspection PyAbstractClass
class EllipticalMassProfile(geometry_profiles.EllipticalProfile, MassProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
    ):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).
        """
        super(EllipticalMassProfile, self).__init__(
            centre=centre, elliptical_comps=elliptical_comps
        )

    @property
    def mass_profile_centres(self):
        if not self.is_mass_sheet:
            return grids.GridCoordinates([self.centre])
        else:
            return []

    def mass_within_circle_in_units(
        self,
        radius: dim.Length,
        unit_mass="angular",
        redshift_object=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
    ):
        """ Integrate the mass profiles's convergence profile to compute the total mass within a circle of \
        specified radius. This is centred on the mass profile.

        The following unit_label for mass can be specified and output:

        - Dimensionless angular unit_label (default) - 'angular'.
        - Solar masses - 'angular' (multiplies the angular mass by the critical surface mass density).

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        unit_mass : str
            The unit_label the mass is returned in {angular, angular}.
        critical_surface_density : float or None
            The critical surface mass density of the strong lens configuration, which converts mass from angulalr \
            unit_label to phsical unit_label (e.g. solar masses).
        """

        if not hasattr(radius, "unit_length"):
            radius = dim.Length(value=radius, unit_length="arcsec")

        if self.unit_length is not radius.unit_length:

            kpc_per_arcsec = cosmology_util.kpc_per_arcsec_from(
                redshift=redshift_object, cosmology=cosmology
            )

            radius = radius.convert(
                unit_length=self.unit_length, kpc_per_arcsec=kpc_per_arcsec
            )

        mass = dim.Mass(
            value=quad(self.mass_integral, a=0.0, b=radius)[0], unit_mass=self.unit_mass
        )

        if unit_mass is "solMass":

            critical_surface_density = cosmology_util.critical_surface_density_between_redshifts_from(
                redshift_0=redshift_object,
                redshift_1=redshift_source,
                cosmology=cosmology,
                unit_length=self.unit_length,
                unit_mass=unit_mass,
            )

        else:

            critical_surface_density = None

        return mass.convert(
            unit_mass=unit_mass, critical_surface_density=critical_surface_density
        )

    def density_between_circular_annuli_in_angular_units(
        self,
        inner_annuli_radius: dim.Length,
        outer_annuli_radius: dim.Length,
        unit_length="arcsec",
        unit_mass="angular",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
    ):
        """Calculate the mass between two circular annuli and compute the density by dividing by the annuli surface
        area.

        The value returned by the mass integral is dimensionless, therefore the density between annuli is returned in \
        unit_label of inverse radius squared. A conversion factor can be specified to convert this to a physical value \
        (e.g. the critical surface mass density).

        Parameters
        -----------
        inner_annuli_radius : float
            The radius of the inner annulus outside of which the density are estimated.
        outer_annuli_radius : float
            The radius of the outer annulus inside of which the density is estimated.
        """
        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (
            np.pi * inner_annuli_radius ** 2.0
        )

        outer_mass = self.mass_within_circle_in_units(
            radius=outer_annuli_radius,
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            unit_mass=unit_mass,
            cosmology=cosmology,
        )

        inner_mass = self.mass_within_circle_in_units(
            radius=inner_annuli_radius,
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            unit_mass=unit_mass,
            cosmology=cosmology,
        )

        return dim.MassOverLength2(
            value=(outer_mass - inner_mass) / annuli_area,
            unit_length=unit_length,
            unit_mass=unit_mass,
        )

    def average_convergence_of_1_radius_in_units(
        self, unit_length="arcsec", redshift_object=None, cosmology=cosmo.Planck15
    ):
        """The radius a critical curve forms for this mass profile, e.g. where the mean convergence is equal to 1.0.

         In case of ellipitical mass profiles, the 'average' critical curve is used, whereby the convergence is \
         rescaled into a circle using the axis ratio.

         This radius corresponds to the Einstein radius of the mass profile, and is a property of a number of \
         mass profiles below.
         """

        if unit_length is "kpc":

            kpc_per_arcsec = cosmology_util.kpc_per_arcsec_from(
                redshift=redshift_object, cosmology=cosmology
            )

        else:

            kpc_per_arcsec = None

        def func(radius, redshift_profile, cosmology):
            radius = dim.Length(radius, unit_length=unit_length)
            return (
                self.mass_within_circle_in_units(
                    unit_mass="angular",
                    radius=radius,
                    redshift_object=redshift_profile,
                    cosmology=cosmology,
                )
                - np.pi * radius ** 2.0
            )

        radius = (
            self.ellipticity_rescale
            * root_scalar(
                func, bracket=[1e-4, 1000.0], args=(redshift_object, cosmology)
            ).root
        )
        radius = dim.Length(radius, unit_length)
        return radius.convert(unit_length=unit_length, kpc_per_arcsec=kpc_per_arcsec)


def psi_from(grid, axis_ratio, core_radius):
    """
    Returns the $\Psi$ term in expressions for the calculation of the deflection of an elliptical isothermal mass
    distribution. This is used in the `Isothermal` and `Chameleon` `MassProfile`'s.

    The expression for Psi is:

    $\Psi = \sqrt(q^2(s^2 + x^2) + y^2)$

    Parameters
    ----------
    grid : grid_like
        The (y,x) coordinates of the grid, in an arrays of shape (total_coordinates, 2)
    axis_ratio : float
        Ratio of profiles ellipse's minor and major axes (b/a)
    core_radius : float
        The radius of the inner core

    Returns
    -------
    float
        The value of the Psi term.

    """
    return np.sqrt(
        np.add(
            np.multiply(
                axis_ratio ** 2.0, np.add(np.square(grid[:, 1]), core_radius ** 2.0)
            ),
            np.square(grid[:, 0]),
        )
    )
