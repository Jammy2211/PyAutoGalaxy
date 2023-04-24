import numpy as np
from scipy.integrate import quad
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.operate.image import OperateImage
from autogalaxy.profiles.geometry_profiles import EllProfile


class LightProfile(EllProfile, OperateImage):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
    ):
        """
        Abstract base class for an elliptical light-profile.

        Each light profile has an analytic equation associated with it that describes its 1D surface brightness.

        Given an input grid of 1D or 2D (y,x) coordinates the light profile can be used to evaluate its surface
        brightness in 1D or as a 2D image.

        Associated with a light profile is a spherical or elliptical geometry, which describes its `centre` of
        emission and ellipticity. Geometric transformations are performed by decorators linked to the **PyAutoArray**
        `geometry` package.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system (see the module
            `autogalaxy -> convert.py` for the convention).
        """
        super().__init__(centre=centre, ell_comps=ell_comps)
        self.intensity = intensity

    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> aa.Array2D:
        """
        Returns the light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates, which may have been
        transformed using the light profile's geometry.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the `LightProfile` evaluated at every (y,x) coordinate on the transformed grid.
        """
        raise NotImplementedError()

    def image_2d_via_radii_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Returns the light profile's 2D image from a 1D grid of coordinates which are the radial distance of each
        coordinate from the light profile `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """
        raise NotImplementedError()

    @aa.grid_dec.grid_1d_to_structure
    def image_1d_from(self, grid: aa.type.Grid1D2DLike) -> aa.type.Grid1D2DLike:
        """
        Returns the light profile's 1D image from a grid of Cartesian coordinates, which may have been
        transformed using the light profile's geometry.

        If a 1D grid is input the image is evaluated every coordinate on the grid. If a 2D grid is input, this is
        converted to a 1D grid by aligning with the major-axis of the light profile's elliptical geometry.

        Internally, this function uses a 2D grid to compute the image, which is mapped to a 1D data structure on return
        via the `grid_1d_to_structure` decorator. This avoids code repetition by ensuring that light profiles only use
        their `image_2d_from()`  function to evaluate their image.

        Parameters
        ----------
        grid
            A 1D or 2D grid of coordinates which are used to evaluate the light profile in 1D.

        Returns
        -------
        image
            The 1D image of the light profile evaluated at every (x,) coordinate on the 1D transformed grid.
        """
        return self.image_2d_from(grid=grid)

    def luminosity_within_circle_from(self, radius: float) -> float:
        """
        Integrate the light profile to compute the total luminosity within a circle of specified radius. This is
        centred on the light profile's `centre`.

        The `intensity` of a light profile is in dimension units, which are given physical meaning when the light
        profile is compared to data with physical units. The luminosity output by this function therefore is also
        dimensionless until compared to data.

        Parameters
        ----------
        radius
            The radius of the circle to compute the dimensionless luminosity within.
        """

        return quad(func=self.luminosity_integral, a=0.0, b=radius)[0]

    def luminosity_integral(self, x: np.ndarray) -> np.ndarray:
        """
        Routine to integrate the luminosity of an elliptical light profile.

        The axis ratio is set to 1.0 for computing the luminosity within a circle

        Parameters
        ----------
        x
            The 1D (x) radial coordinates where the luminosity integral is evaluated.
        """
        return 2 * np.pi * x * self.image_2d_via_radii_from(x)

    @property
    def half_light_radius(self) -> float:
        if hasattr(self, "effective_radius"):
            return self.effective_radius

    @property
    def _intensity(self):
        return self.intensity
