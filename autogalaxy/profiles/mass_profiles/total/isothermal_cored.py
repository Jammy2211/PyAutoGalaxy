import copy
import numpy as np
from scipy.integrate import quad
from scipy import special
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.abstract.abstract import MassProfile


class EllIsothermalCored(EllPowerLawCored):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored elliptical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        einstein_radius
            The arc-second Einstein radius.
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )


class SphIsothermalCored(SphPowerLawCored):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored spherical isothermal density distribution, which is equivalent to the elliptical power-law
        density distribution for the value slope: float = 2.0

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(
            centre=centre,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )
