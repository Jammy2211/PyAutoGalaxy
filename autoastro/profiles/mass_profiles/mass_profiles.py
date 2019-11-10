import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from astropy import cosmology as cosmo
from skimage import measure

import autofit as af
from autoarray.structures import grids
from autoastro import lensing
from autoastro import dimensions as dim
from autofit.tools import text_util
from autoastro.profiles import geometry_profiles


class MassProfile(lensing.LensingObject):
    def summarize_in_units(
        self,
        radii,
        prefix="",
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        whitespace=80,
        **kwargs,
    ):
        return ["Mass Profile = {}\n".format(self.__class__.__name__)]


# noinspection PyAbstractClass
class EllipticalMassProfile(geometry_profiles.EllipticalProfile, MassProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
    ):
        """
        Abstract class for elliptical mass profiles.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        axis_ratio : float
            Ellipse's minor-to-major axis ratio (b/a)
        phi : float
            Rotation angle of profile's ellipse counter-clockwise from positive x-axis
        """
        super(EllipticalMassProfile, self).__init__(
            centre=centre, axis_ratio=axis_ratio, phi=phi
        )
        self.axis_ratio = axis_ratio
        self.phi = phi

    def mass_within_circle_in_units(
        self,
        radius: dim.Length,
        unit_mass="angular",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs,
    ):
        """ Integrate the mass profiles's convergence profile to compute the total mass within a circle of \
        specified radius. This is centred on the mass profile.

        The following units for mass can be specified and output:

        - Dimensionless angular units (default) - 'angular'.
        - Solar masses - 'angular' (multiplies the angular mass by the critical surface mass density).

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        unit_mass : str
            The units the mass is returned in (angular | angular).
        critical_surface_density : float or None
            The critical surface mass density of the strong lens configuration, which converts mass from angulalr \
            units to phsical units (e.g. solar masses).
        """

        critical_surface_density = (
            kwargs["critical_surface_density"]
            if "critical_surface_density" in kwargs
            else None
        )

        mass = dim.Mass(
            value=quad(self.mass_integral, a=0.0, b=radius)[0], unit_mass=self.unit_mass
        )

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
        **kwargs,
    ):
        """Calculate the mass between two circular annuli and compute the density by dividing by the annuli surface
        area.

        The value returned by the mass integral is dimensionless, therefore the density between annuli is returned in \
        units of inverse radius squared. A conversion factor can be specified to convert this to a physical value \
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
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            unit_mass=unit_mass,
            cosmology=cosmology,
        )

        inner_mass = self.mass_within_circle_in_units(
            radius=inner_annuli_radius,
            redshift_profile=redshift_profile,
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
        self,
        unit_length="arcsec",
        redshift_profile=None,
        cosmology=cosmo.Planck15,
        **kwargs,
    ):
        """The radius a critical curve forms for this mass profile, e.g. where the mean convergence is equal to 1.0.

         In case of ellipitical mass profiles, the 'average' critical curve is used, whereby the convergence is \
         rescaled into a circle using the axis ratio.

         This radius corresponds to the Einstein radius of the mass profile, and is a property of a number of \
         mass profiles below.
         """

        kpc_per_arcsec = (
            kwargs["kpc_per_arcsec"] if "kpc_per_arcsec" in kwargs else None
        )

        def func(radius, redshift_profile, cosmology):
            radius = dim.Length(radius, unit_length=unit_length)
            return (
                self.mass_within_circle_in_units(
                    unit_mass="angular",
                    radius=radius,
                    redshift_profile=redshift_profile,
                    cosmology=cosmology,
                )
                - np.pi * radius ** 2.0
            )

        radius = (
            self.ellipticity_rescale
            * root_scalar(
                func, bracket=[1e-4, 1000.0], args=(redshift_profile, cosmology)
            ).root
        )
        radius = dim.Length(radius, unit_length)
        return radius.convert(unit_length=unit_length, kpc_per_arcsec=kpc_per_arcsec)

    def summarize_in_units(
        self,
        radii,
        prefix="",
        whitespace=80,
        unit_length="arcsec",
        unit_mass="solMass",
        redshift_profile=None,
        redshift_source=None,
        cosmology=cosmo.Planck15,
        **kwargs,
    ):
        summary = super().summarize_in_units(
            radii=radii,
            prefix="",
            unit_length=unit_length,
            unit_mass=unit_mass,
            redshift_profile=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        einstein_radius = self.einstein_radius_in_units(
            unit_length=unit_length,
            redshift_object=redshift_profile,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        summary += [
            text_util.label_value_and_unit_string(
                label=prefix + "einstein_radius",
                value=einstein_radius,
                unit=unit_length,
                whitespace=whitespace,
            )
        ]

        einstein_mass = self.einstein_mass_in_units(
            unit_mass=unit_mass,
            redshift_object=redshift_profile,
            redshift_source=redshift_source,
            cosmology=cosmology,
            kwargs=kwargs,
        )

        summary += [
            text_util.label_value_and_unit_string(
                label=prefix + "einstein_mass",
                value=einstein_mass,
                unit=unit_mass,
                whitespace=whitespace,
            )
        ]

        for radius in radii:
            mass = self.mass_within_circle_in_units(
                unit_mass=unit_mass,
                radius=radius,
                redshift_profile=redshift_profile,
                redshift_source=redshift_source,
                cosmology=cosmology,
                kwargs=kwargs,
            )

            summary += [
                text_util.within_radius_label_value_and_unit_string(
                    prefix=prefix + "mass",
                    radius=radius,
                    unit_length=unit_length,
                    value=mass,
                    unit_value=unit_mass,
                    whitespace=whitespace,
                )
            ]

        return summary

    @property
    def ellipticity_rescale(self):
        return NotImplementedError()
