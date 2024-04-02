import numpy as np
from typing import Optional, Tuple

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.decorators import (
    check_operated_only,
)


class Chameleon(LightProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.05,
    ):
        """
        The elliptical Chameleon light profile.

        This light profile closely approximes the Elliptical Sersic light profile, by representing it as two cored
        elliptical isothermal profiles. This is convenient for lensing calculations, because the deflection angles of
        an isothermal profile can be evaluated analyticially efficiently.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        core_radius_0
            The core size of the first elliptical cored Isothermal profile.
        core_radius_1
            The core size of the second elliptical cored Isothermal profile.
        """

        super().__init__(centre=centre, ell_comps=ell_comps, intensity=intensity)
        self.core_radius_0 = core_radius_0
        self.core_radius_1 = core_radius_1

    @property
    def axis_ratio(self) -> float:
        """
        The elliptical isothermal mass profile deflection angles break down for perfectly spherical systems where
        `axis_ratio=1.0`, thus we remove these solutions.
        """
        axis_ratio = super().axis_ratio
        return axis_ratio if axis_ratio < 0.99999 else 0.99999

    def image_2d_via_radii_from(self, grid_radii: np.ndarray) -> np.ndarray:
        """
        Returns the 2D image of the Sersic light profile from a grid of coordinates which are the radial distances of
        each coordinate from the its `centre`.

        Parameters
        ----------
        grid_radii
            The radial distances from the centre of the profile, for each coordinate on the grid.
        """

        axis_ratio_factor = (1.0 + self.axis_ratio) ** 2.0

        return np.multiply(
            self._intensity / (1 + self.axis_ratio),
            np.add(
                np.divide(
                    1.0,
                    np.sqrt(
                        np.add(
                            np.square(grid_radii),
                            (4.0 * self.core_radius_0**2.0) / axis_ratio_factor,
                        )
                    ),
                ),
                -np.divide(
                    1.0,
                    np.sqrt(
                        np.add(
                            np.square(grid_radii),
                            (4.0 * self.core_radius_1**2.0) / axis_ratio_factor,
                        )
                    ),
                ),
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
        Returns the Chameleon light profile's 2D image from a 2D grid of Cartesian (y,x) coordinates.

        If the coordinates have not been transformed to the profile's geometry (e.g. translated to the
        profile `centre`), this is performed automatically.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates in the original reference frame of the grid.

        Returns
        -------
        image
            The image of the Chameleon evaluated at every (y,x) coordinate on the transformed grid.
        """
        return self.image_2d_via_radii_from(self.elliptical_radii_grid_from(grid))


class ChameleonSph(Chameleon):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        core_radius_0: float = 0.01,
        core_radius_1: float = 0.05,
    ):
        """
        The spherical Chameleon light profile.

        This light profile closely approximes the Elliptical Sersic light profile, by representing it as two cored
        elliptical isothermal profiles. This is convenient for lensing calculations, because the deflection angles of
        an isothermal profile can be evaluated analyticially efficiently.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        core_radius_0
            The core size of the first elliptical cored Isothermal profile.
        core_radius_1
            The core size of the second elliptical cored Isothermal profile.
        """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            core_radius_0=core_radius_0,
            core_radius_1=core_radius_1,
        )
