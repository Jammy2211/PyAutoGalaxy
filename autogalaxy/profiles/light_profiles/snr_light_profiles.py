import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.light_profiles import light_profiles as lp
from scipy.optimize import root_scalar


class SNRCalc:
    def __init__(self, signal_to_noise_ratio: float = 10.0):

        self.signal_to_noise_ratio = signal_to_noise_ratio

    def image_2d_from(self, grid: aa.type.Grid2DLike) -> aa.Array2D:
        """
        Abstract method for obtaining intensity at a grid of Cartesian (y,x) coordinates.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image : aa.Array2D
            The image of the `LightProfile` evaluated at every (y,x) coordinate on the grid.
        """
        raise NotImplementedError()

    def set_intensity_from(
        self,
        grid: aa.type.Grid2DLike,
        exposure_time: float,
        background_sky_level: float = 0.0,
    ):

        self.intensity = 1.0

        background_sky_level_counts = background_sky_level * exposure_time

        brightest_value = np.max(self.image_2d_from(grid=grid))

        def func(intensity_factor):

            signal = intensity_factor * brightest_value * exposure_time
            noise = np.sqrt(signal + background_sky_level_counts)

            return signal / noise - self.signal_to_noise_ratio

        intensity_factor = root_scalar(func, bracket=[1.0e-8, 1.0e8]).root

        self.intensity *= intensity_factor


class EllSersic(lp.EllSersic, SNRCalc):
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

        SNRCalc.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)
