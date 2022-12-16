import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)


class Gaussian(LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 1.0,
    ):
        """
        The elliptical Gaussian light profile.

        The intensity distribution of the profile is given by:

        .. math:: I(\\xi) = I \exp (-0.5 \\xi / (\sigma / q^{0.5}))^2

        Where \\xi are elliptical coordinates calculated according to :class: SphProfile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        sigma
            The sigma value of the Gaussian, corresponding to ~ 1 / sqrt(2 log(2)) the full width half maximum.
        """
        super().__init__(centre=centre, ell_comps=ell_comps, intensity=intensity)
        self.sigma = sigma

    def image_2d_via_radii_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Gaussian light profile from a grid of coordinates which are the radial distance of
        each coordinate from the its `centre`.

        Note: sigma is divided by sqrt(q) here.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        return np.multiply(
            self._intensity,
            np.exp(
                -0.5
                * np.square(
                    np.divide(grid_radii, self.sigma / np.sqrt(self.axis_ratio))
                )
            ),
        )

    @aa.grid_dec.grid_2d_to_structure
    @check_operated_only
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> np.ndarray:
        """
        Returns the Gaussian light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Gaussian evaluated at every (y,x) coordinate on the transformed grid.
        """

        return self.image_2d_via_radii_from(self.eccentric_radii_grid_from(grid))


class GaussianSph(Gaussian):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 1.0,
    ):
        """
        The spherical Gaussian light profile.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        sigma
            The sigma value of the Gaussian, corresponding to ~ 1 / sqrt(2 log(2)) the full width half maximum.
        """
        super().__init__(
            centre=centre, ell_comps=(0.0, 0.0), intensity=intensity, sigma=sigma
        )
