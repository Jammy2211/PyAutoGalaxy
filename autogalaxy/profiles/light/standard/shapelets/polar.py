import numpy as np
from scipy.special import factorial, genlaguerre
from typing import Optional, Tuple

import autoarray as aa


from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)
from autogalaxy.profiles.light.standard.shapelets.abstract import AbstractShapelet


class ShapeletPolar(AbstractShapelet):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Polar (r,theta) grid of coordinates.

        Shapelets are defined according to:

          https://arxiv.org/abs/astro-ph/0105178

        Shapelets are described in the context of strong lens modeling in:

          https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T/abstract

        Parameters
        ----------
        n
            The n order of the shapelets basis function.
        m
            The m order of the shapelets basis function in the x-direction.
        centre
            The (y,x) arc-second coordinates of the profile (shapelet) centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        self.n = n
        self.m = m

        super().__init__(
            centre=centre, ell_comps=ell_comps, beta=beta, intensity=intensity
        )

    @property
    def coefficient_tag(self) -> str:
        return f"n_{self.n}_m_{self.m}"

    @aa.over_sample
    @aa.grid_dec.to_array
    @check_operated_only
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None, **kwargs
    ) -> np.ndarray:
        """
        Returns the Polar Shapelet light profile's 2D image from a 2D grid of Polar (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Polar Shapelet evaluated at every (y,x) coordinate on the transformed grid.
        """

        laguerre = genlaguerre(n=(self.n - np.abs(self.m)) / 2.0, alpha=np.abs(self.m))

        const = (
            ((-1) ** ((self.n - np.abs(self.m)) // 2))
            * np.sqrt(
                factorial((self.n - np.abs(self.m)) // 2)
                / factorial((self.n + np.abs(self.m)) // 2)
            )
            / self.beta
            / np.sqrt(np.pi)
        )

        rsq = (grid[:, 0] ** 2 + grid[:, 1] ** 2) / self.beta**2
        theta = np.arctan2(grid[:, 1], grid[:, 0])
        radial = rsq ** (abs(self.m / 2.0)) * np.exp(-rsq / 2.0) * laguerre(rsq)

        if self.m == 0:
            azimuthal = 1
        elif self.m > 0:
            azimuthal = np.sin((-1) * self.m * theta)
        else:
            azimuthal = np.cos((-1) * self.m * theta)

        return const * radial * azimuthal


class ShapeletPolarSph(ShapeletPolar):
    def __init__(
        self,
        n: int,
        m: int,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 1.0,
        beta: float = 1.0,
    ):
        """
        Shapelets where the basis function is defined according to a Polar (r,theta) grid of coordinates.

        Shapelets are defined according to:

          https://arxiv.org/abs/astro-ph/0105178

        Shapelets are described in the context of strong lens modeling in:

          https://ui.adsabs.harvard.edu/abs/2016MNRAS.457.3066T/abstract

        Parameters
        ----------
        n_y
            The order of the shapelets basis function in the y-direction.
        n_x
            The order of the shapelets basis function in the x-direction.
        centre
            The (y,x) arc-second coordinates of the profile (shapelet) centre.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        beta
            The characteristic length scale of the shapelet basis function, defined in arc-seconds.
        """

        super().__init__(
            n=n,
            m=m,
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            beta=beta,
        )
