from typing import Tuple

from autogalaxy.profiles.mass.total.power_law_core import PowerLawCore
from autogalaxy.profiles.mass.total.power_law_core import PowerLawCoreSph


class IsothermalCore(PowerLawCore):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
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
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            einstein_radius=einstein_radius,
            slope=2.0,
            core_radius=core_radius,
        )


class IsothermalCoreSph(PowerLawCoreSph):
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
