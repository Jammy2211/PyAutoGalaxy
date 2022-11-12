from astropy import cosmology as cosmo
from astropy import units
from colossus.cosmology import cosmology as col_cosmology
from colossus.halo.concentration import concentration as col_concentration
import copy
import inspect
import numpy as np
from scipy import LowLevelCallable
from scipy import special
from scipy.integrate import quad
from scipy.optimize import fsolve
import warnings
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass_profiles.dark.abstract import DarkProfile
from autogalaxy.profiles.mass_profiles import MassProfile
from autogalaxy.cosmology.lensing import LensingCosmology
from autogalaxy.cosmology.wrap import Planck15

from autogalaxy.profiles.mass_profiles.mass_profiles import (
    MassProfileMGE,
    MassProfileCSE,
)

from autogalaxy import exc


class SphNFWTruncatedMCRDuffy(SphNFWTruncated):
    """
    This function only applies for the lens configuration as follows:
    Cosmology: FlatLamdaCDM
    H0 = 70.0 km/sec/Mpc
    Omega_Lambda = 0.7
    Omega_m = 0.3
    Redshfit of Main Lens: 0.6
    Redshift of Source: 2.5
    A truncated NFW halo at z = 0.6 with tau = 2.0
    """

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        """
        Input m200: The m200 of the NFW part of the corresponding tNFW part. Unit: M_sun.
        """

        self.mass_at_200 = mass_at_200

        kappa_s, scale_radius, radius_at_200 = kappa_s_and_scale_radius_for_duffy(
            mass_at_200=mass_at_200,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=2.0 * radius_at_200,
        )


class SphNFWTruncatedMCRScatterLudlow(SphNFWTruncated):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        scatter_sigma: float = 0.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):

        self.mass_at_200 = mass_at_200
        self.scatter_sigma = scatter_sigma
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        kappa_s, scale_radius, radius_at_200 = kappa_s_and_scale_radius_for_ludlow(
            mass_at_200=mass_at_200,
            scatter_sigma=scatter_sigma,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(
            centre=centre,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
            truncation_radius=2.0 * radius_at_200,
        )


class SphNFWTruncatedMCRLudlow(SphNFWTruncatedMCRScatterLudlow):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        """
        Input m200: The m200 of the NFW part of the corresponding tNFW part. Unit: M_sun.
        """

        super().__init__(
            centre=centre,
            mass_at_200=mass_at_200,
            scatter_sigma=0.0,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )
