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


class SphNFWMCRDuffy(SphNFW):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):

        self.mass_at_200 = mass_at_200
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        kappa_s, scale_radius, radius_at_200 = kappa_s_and_scale_radius_for_duffy(
            mass_at_200=mass_at_200,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(centre=centre, kappa_s=kappa_s, scale_radius=scale_radius)

    def with_new_normalization(self, normalization):

        raise NotImplementedError()


class EllNFWMCRLudlow(EllNFW):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):

        self.mass_at_200 = mass_at_200
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        kappa_s, scale_radius, radius_at_200 = kappa_s_and_scale_radius_for_ludlow(
            mass_at_200=mass_at_200,
            scatter_sigma=0.0,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            kappa_s=kappa_s,
            scale_radius=scale_radius,
        )


class SphNFWMCRLudlow(SphNFWMCRScatterLudlow):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):

        super().__init__(
            centre=centre,
            mass_at_200=mass_at_200,
            scatter_sigma=0.0,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )
