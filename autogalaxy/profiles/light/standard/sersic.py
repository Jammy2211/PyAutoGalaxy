import jax.numpy as jnp
import numpy as np

from numpy import seterr
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)


class AbstractSersic(LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        Abstract base class for elliptical Sersic light profiles.

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
            The circular radius containing half the light of this light profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        """
        super().__init__(centre=centre, ell_comps=ell_comps, intensity=intensity)
        self.effective_radius = effective_radius
        self.sersic_index = sersic_index

    @property
    def elliptical_effective_radius(self) -> float:
        """
        The `effective_radius` of a Sersic light profile is defined as the circular effective radius, which is the
        radius within which a circular aperture contains half the profile's total integrated light.

        For elliptical systems, this will not robustly capture the light profile's elliptical shape.

        The elliptical effective radius instead describes the major-axis radius of the ellipse containing
        half the light, and may be more appropriate for highly flattened systems like disk galaxies.
        """
        return self.effective_radius / jnp.sqrt(self.axis_ratio)

    @property
    def sersic_constant(self) -> float:
        """
        A parameter derived from Sersic index which ensures that effective radius contains 50% of the profile's
        total integrated light.
        """
        return (
            (2 * self.sersic_index)
            - (1.0 / 3.0)
            + (4.0 / (405.0 * self.sersic_index))
            + (46.0 / (25515.0 * self.sersic_index**2))
            + (131.0 / (1148175.0 * self.sersic_index**3))
            - (2194697.0 / (30690717750.0 * self.sersic_index**4))
        )

    def image_2d_via_radii_from(self, radius: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        return self._intensity * jnp.exp(
            -self.sersic_constant
            * (((radius / self.effective_radius) ** (1.0 / self.sersic_index)) - 1)
        )


class Sersic(AbstractSersic, LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        The elliptical Sersic light profile.

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
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, **kwargs) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        seterr(all="ignore")
        return jnp.multiply(
            self._intensity,
            jnp.exp(
                jnp.multiply(
                    -self.sersic_constant,
                    jnp.add(
                        jnp.power(
                            jnp.divide(grid_radii.array, self.effective_radius),
                            1.0 / self.sersic_index,
                        ),
                        -1,
                    ),
                )
            ),
        )

    @aa.over_sample
    @aa.grid_dec.to_array
    @check_operated_only
    @aa.grid_dec.transform
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None, **kwargs
    ) -> aa.Array2D:
        """
        Returns the Sersic light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Sersic evaluated at every (y,x) coordinate on the transformed grid.
        """

        grid_radii = self.eccentric_radii_grid_from(grid=grid, **kwargs)

        return self.image_2d_via_radii_from(grid_radii=grid_radii, **kwargs)


class SersicSph(Sersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        The spherical Sersic light profile.

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
            Controls the concentration of the of the light profile.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )
