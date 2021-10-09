from typing import Tuple

from autogalaxy.profiles.light_profiles import light_profiles as lp


class EllSersic(lp.EllSersic):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        The elliptical Sersic light profile.

        Parameters
        ----------
        signal_to_noise_ratio
            The signal to noise of the light profile when it is used to simulate strong lens imaging.
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*angle) and ellip_x = fac * cos(2*angle).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index : Int
            Controls the concentration of the of the profile (lower value -> less concentrated, \
            higher value -> more concentrated).
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=0.0,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )

        self.signal_to_noise_ratio = signal_to_noise_ratio

