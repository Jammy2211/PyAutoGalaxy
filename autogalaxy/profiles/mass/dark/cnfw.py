import numpy as np

from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile

class cNFW(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        kappa_s: float = 0.05,
        scale_radius: float = 1.0,
        core_radius: float = 0.5,
    ):
        """
                Represents a spherical cored NFW density distribution

                Parameters
                ----------
                centre
                    The (y,x) arc-second coordinates of the profile centre.
                kappa_s
                    The overall normalization of the dark matter halo \|
                    (kappa_s = (rho_0 * scale_radius)/lensing_critical_density)
                scale_radius
                    The cored NFW scale radius `theta_s`, as an angle on the sky in arcseconds.
                core_radius
                    The cored NFW core radius `theta_c`, as an angle on the sky in arcseconds.
                """

        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0))

        self.kappa_s = kappa_s
        self.scale_radius = scale_radius
        self.core_radius = core_radius


    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        The input grid of (y,x) coordinates are transformed to a coordinate system centred on the profile centre with
        and rotated based on the position angle defined from its `ell_comps` (this is described fully below).

        The numerical backend can be selected via the ``xp`` argument, allowing this
        method to be used with both NumPy and JAX (e.g. inside ``jax.jit``-compiled
        code). This is described fully later in this example.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        xp
            The numerical backend to use, either `numpy` or `jax.numpy`.
        """
        theta = self.radial_grid_from(grid=grid, xp=xp, **kwargs).array
        theta = xp.maximum(theta, 1e-8)
        
        factor = (
            4.0
            * self.kappa_s
            * self.scale_radius**2
        )

        deflection_r = (
            factor
            * (self.F_func(theta, self.scale_radius, xp=xp) - self.F_func(theta, self.core_radius, xp=xp)
               - (self.scale_radius - self.core_radius) * self.dev_F_func(theta, self.scale_radius, xp=xp)
            )
            / (theta * (self.scale_radius - self.core_radius)**2)
        )


        return self._cartesian_grid_via_radial_from(
            grid=grid,
            radius=deflection_r,
            xp=xp,
            **kwargs,
        )

    def F_func(self, theta, radius, xp=np):

        F = theta * 0.0

        # theta == 0
        mask0 = theta == 0

        # theta < radius
        mask1 = (theta > 0) & (theta < radius)

        # theta > radius
        mask2 = theta > radius

        F = xp.where(
            mask0,
            (
                0.0
            ),
            F,
        )

        F = xp.where(
            mask1,
            (
                    radius / 2 * xp.log(2 * radius / theta)
                    - xp.sqrt(radius ** 2 - theta ** 2)
                    * xp.arctanh(xp.sqrt((radius - theta) / (radius + theta)))
            ),
            F,
        )

        F = xp.where(
            mask2,
            (
                    radius / 2 * xp.log(2 * radius / theta)
                    + xp.sqrt(theta ** 2 - radius ** 2)
                    * xp.arctan(xp.sqrt((theta - radius) / (theta + radius)))
            ),
            F,
        )

        return 2 * radius * F

    def dev_F_func(self, theta, radius, xp=np):

        dev_F = theta * 0.0

        mask0 = theta == 0
        mask1 = (theta > 0) & (theta < radius)
        mask2 = theta == radius
        mask3 = theta > radius

        dev_F = xp.where(
            mask0,
            (
                0.0
            ),
            dev_F,
        )

        dev_F = xp.where(
            mask1,
            (
                    radius * xp.log(2 * radius / theta)
                    - (2 * radius ** 2 - theta ** 2) / xp.sqrt(radius ** 2 - theta ** 2)
                    * xp.arctanh(xp.sqrt((radius - theta) / (radius + theta)))
            ),
            dev_F,
        )

        dev_F = xp.where(
            mask2,
            radius * (xp.log(2) - 1 / 2),
            dev_F,
        )

        dev_F = xp.where(
            mask3,
            (
                    radius * xp.log(2 * radius / theta)
                    + (theta ** 2 - 2 * radius ** 2) / xp.sqrt(theta ** 2 - radius ** 2)
                    * xp.arctan(xp.sqrt((theta - radius) / (theta + radius)))
            ),
            dev_F,
        )

        return 2 * dev_F

    # def F_func(self, theta, radius, xp=np):
    #     if theta == 0:
    #         F = 0
    #     elif theta < radius:
    #         F = (radius / 2 * xp.log(2 * radius / theta) - xp.sqrt(radius ** 2 - theta ** 2)
    #              * xp.arctanh(xp.sqrt((radius - theta) / (radius + theta)))
    #              )
    #     else:
    #         F = (radius / 2 * xp.log(2 * radius / theta) + xp.sqrt(theta ** 2 - radius ** 2)
    #              * xp.arctan(xp.sqrt((theta - radius) / (theta + radius)))
    #              )
    #     return 2 * radius * F

    # def dev_F_func(self, theta, radius, xp=np):
    #     if theta == 0:
    #         dev_F = 0
    #     elif theta < radius:
    #         dev_F = (radius * xp.log(2 * radius / theta)
    #                  - (2 * radius ** 2 - theta ** 2) / xp.sqrt(radius ** 2 - theta ** 2)
    #                  * xp.arctanh(xp.sqrt((radius - theta) / (radius + theta)))
    #                  )
    #     elif theta == radius:
    #         dev_F = (radius * (xp.log(2) - 1 / 2))
    #     else:
    #         dev_F = (radius * xp.log(2 * radius / theta)
    #                  + (theta ** 2 - 2 * radius ** 2) / xp.sqrt(theta ** 2 - radius ** 2)
    #                  * xp.arctan(xp.sqrt((theta - radius) / (theta + radius)))
    #                  )
    #     return 2 * dev_F