from typing import Tuple

from autogalaxy.profiles.light_profiles.linear.abstract import LightProfileLinear

from autogalaxy.profiles.light_profiles import base as lp
from autogalaxy.profiles import light_and_mass_profiles as lmp


class EllGaussian(lp.EllGaussian, LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 1.0,
    ):
        """
        The elliptical Gaussian light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        sigma
            The sigma value of the Gaussian, corresponding to ~ 1 / sqrt(2 log(2)) the full width half maximum.
        """

        super().__init__(
            centre=centre, elliptical_comps=elliptical_comps, intensity=1.0, sigma=sigma
        )

    @property
    def lp_cls(self):
        return lp.EllGaussian

    @property
    def lmp_cls(self):
        return lmp.EllGaussian
