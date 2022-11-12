import inspect
import numpy as np
from typing import Dict, List, Optional, Tuple

from autoconf import cached_property
import autoarray as aa
import autofit as af

from autogalaxy.profiles.light_profiles.light_profiles_operated import (
    LightProfileOperated,
)

from autogalaxy.profiles.light_profiles import light_profiles as lp
from autogalaxy.profiles import light_and_mass_profiles as lmp

from autogalaxy import exc


class EllSersicCore(lp.EllSersicCore, LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        gamma: float = 0.25,
        alpha: float = 3.0,
    ):
        """
        The elliptical cored-Sersic light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        radius_break
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        gamma
            The logarithmic power-law slope of the inner core profiles
        alpha
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            alpha=alpha,
            gamma=gamma,
        )

    @property
    def lp_cls(self):
        return lp.EllSersicCore

    def parameters_dict_from(self, intensity: float) -> Dict[str, float]:
        """
        Returns a dictionary of the parameters of the linear light profile with the `intensity` added.

        This `intenisty` will likely have come from the value inferred via the linear inversion.

        Parameters
        ----------
        intensity
            Overall intensity normalisation of the not linear light profile that is created (units are dimensionless
            and derived from the data the light profile's image is compared too, which is expected to be electrons
            per second).
        """
        parameters_dict = vars(self)

        args = inspect.getfullargspec(self.lp_cls.__init__).args
        args.remove("self")

        parameters_dict = {key: parameters_dict[key] for key in args}
        parameters_dict["intensity_break"] = intensity

        return parameters_dict
