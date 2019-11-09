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

    @dim.convert_units_to_input_units
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
            redshift_profile=redshift_profile,
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
            redshift_profile=redshift_profile,
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
