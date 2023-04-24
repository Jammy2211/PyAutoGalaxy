import numpy as np
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.stellar.sersic import Sersic


class SersicCore(Sersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical cored-Sersic light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        radius_break
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity
            The intensity at the break radius.
        gamma
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """

        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )

        self.radius_break = radius_break
        self.intensity = intensity
        self.alpha = alpha
        self.gamma = gamma

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        return self.deflections_2d_via_mge_from(grid=grid)

    def image_2d_via_radii_from(self, grid_radii: np.ndarray):
        """
        Calculate the intensity of the cored-Sersic light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.
        """
        return np.multiply(
            np.multiply(
                self.intensity_prime,
                np.power(
                    np.add(
                        1,
                        np.power(np.divide(self.radius_break, grid_radii), self.alpha),
                    ),
                    (self.gamma / self.alpha),
                ),
            ),
            np.exp(
                np.multiply(
                    -self.sersic_constant,
                    (
                        np.power(
                            np.divide(
                                np.add(
                                    np.power(grid_radii, self.alpha),
                                    (self.radius_break**self.alpha),
                                ),
                                (self.effective_radius**self.alpha),
                            ),
                            (1.0 / (self.alpha * self.sersic_index)),
                        )
                    ),
                )
            ),
        )

    def decompose_convergence_via_mge(self):
        radii_min = self.effective_radius / 50.0
        radii_max = self.effective_radius * 20.0

        def core_sersic_2D(r):
            return (
                self.mass_to_light_ratio
                * self.intensity_prime
                * (1.0 + (self.radius_break / r) ** self.alpha)
                ** (self.gamma / self.alpha)
                * np.exp(
                    -self.sersic_constant
                    * (
                        (r**self.alpha + self.radius_break**self.alpha)
                        / self.effective_radius**self.alpha
                    )
                    ** (1.0 / (self.sersic_index * self.alpha))
                )
            )

        return self._decompose_convergence_via_mge(
            func=core_sersic_2D, radii_min=radii_min, radii_max=radii_max
        )

    @property
    def intensity_prime(self):
        """Overall intensity normalisation in the rescaled Core-Sersic light profiles (electrons per second)"""
        return (
            self.intensity
            * (2.0 ** (-self.gamma / self.alpha))
            * np.exp(
                self.sersic_constant
                * (
                    ((2.0 ** (1.0 / self.alpha)) * self.radius_break)
                    / self.effective_radius
                )
                ** (1.0 / self.sersic_index)
            )
        )


class SersicCoreSph(SersicCore):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
    ):
        """
        The elliptical cored-Sersic light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        radius_break
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity
            The intensity at the break radius.
        gamma
            The logarithmic power-law slope of the inner core profiles
        alpha :
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity=intensity,
            gamma=gamma,
            alpha=alpha,
        )
        self.radius_break = radius_break
        self.intensity = intensity
        self.alpha = alpha
        self.gamma = gamma
