import numpy as np

from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.stellar.abstract import StellarProfile


class Gaussian(MassProfile, StellarProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        sigma: float = 1.0,
        mass_to_light_ratio: float = 1.0,
    ):
        """
        The elliptical Gaussian light profile.

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
            The sigma value of the Gaussian.
        """

        super(Gaussian, self).__init__(centre=centre, ell_comps=ell_comps)
        super(MassProfile, self).__init__(centre=centre, ell_comps=ell_comps)
        self.mass_to_light_ratio = mass_to_light_ratio
        self.intensity = intensity
        self.sigma = sigma


    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        return self.deflections_2d_via_analytic_from(grid=grid, xp=xp, **kwargs)

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_2d_via_analytic_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        deflections = (
            self.mass_to_light_ratio
            * self.intensity
            * self.sigma
            * xp.sqrt((2 * xp.pi) / (1.0 - self.axis_ratio(xp) ** 2.0))
            * self.zeta_from(grid=grid, xp=xp)
        )

        return self.rotated_grid_from_reference_frame_from(
            xp.multiply(
                1.0, xp.vstack((-1.0 * xp.imag(deflections), xp.real(deflections))).T
            ),
            xp=xp,
        )

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_2d_via_integral_from(
        self, grid: aa.type.Grid2DLike, xp=np, **kwargs
    ):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        Note: sigma is divided by sqrt(q) here.

        """
        from scipy.integrate import quad

        def calculate_deflection_component(npow, index):
            deflection_grid = np.array(self.axis_ratio(xp) * grid.array[:, index])

            for i in range(grid.shape[0]):
                deflection_grid[i] *= (
                    self.intensity
                    * self.mass_to_light_ratio
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid.array[i, 0],
                            grid.array[i, 1],
                            npow,
                            self.axis_ratio(xp),
                            self.sigma / xp.sqrt(self.axis_ratio(xp)),
                        ),
                    )[0]
                )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotated_grid_from_reference_frame_from(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T), xp=xp
        )

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, sigma, xp=np):
        _eta_u = xp.sqrt(axis_ratio) * xp.sqrt(
            (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
        )

        return xp.exp(-0.5 * xp.square(xp.divide(_eta_u, sigma))) / (
            (1 - (1 - axis_ratio**2) * u) ** (npow + 0.5)
        )

    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(
            self.eccentric_radii_grid_from(grid=grid, xp=xp, **kwargs)
        )

    def convergence_func(self, grid_radius: float) -> float:
        return self.mass_to_light_ratio * self.image_2d_via_radii_from(grid_radius)

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, xp=np, **kwargs):
        return xp.zeros(shape=grid.shape[0])

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, xp=np):
        """Calculate the intensity of the Gaussian light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.

        Note: sigma is divided by sqrt(q) here.
        """
        return xp.multiply(
            self.intensity,
            xp.exp(
                -0.5
                * xp.square(
                    xp.divide(
                        grid_radii.array, self.sigma / xp.sqrt(self.axis_ratio(xp))
                    )
                )
            ),
        )

    def axis_ratio(self, xp=np):
        axis_ratio = super().axis_ratio(xp=xp)
        return xp.where(axis_ratio < 0.9999, axis_ratio, 0.9999)

    def zeta_from(self, grid: aa.type.Grid2DLike, xp=np):
        q = self.axis_ratio(xp)
        q2 = q ** 2.0

        ind_pos_y = grid.array[:, 0] >=0
        shape_grid = np.shape(grid)
        output_grid = np.zeros((shape_grid[0]), dtype=np.complex128)

        scale_factor = q / (self.sigma * xp.sqrt(2.0 * (1.0 - q2)))

        xs_0 = grid.array[:, 1][ind_pos_y] * scale_factor
        ys_0 = grid.array[:, 0][ind_pos_y] * scale_factor
        xs_1 = grid.array[:, 1][~ind_pos_y] * scale_factor
        ys_1 = -grid.array[:, 0][~ind_pos_y] * scale_factor

        z1_0 = xs_0 + 1j * ys_0
        z2_0 = q * xs_0 + 1j * ys_0 / q
        z1_1 = xs_1 + 1j * ys_1
        z2_1 = q * xs_1 + 1j * ys_1 / q

        exp_term_0 = xp.exp(-(xs_0 ** 2) * (1.0 - q2) - ys_0 ** 2 * (1.0 / q2 - 1.0))
        exp_term_1 = xp.exp(-(xs_1 ** 2) * (1.0 - q2) - ys_1 ** 2 * (1.0 / q2 - 1.0))

        if xp == np:
            from scipy.special import wofz

            output_grid[ind_pos_y] = -1j * (wofz(z1_0) - exp_term_0 * wofz(z2_0))
            output_grid[~ind_pos_y] = xp.conj(-1j * (wofz(z1_1) - exp_term_1 * wofz(z2_1)))

        else:
            output_grid[ind_pos_y] = -1j * (self.wofz(z1_0, xp=xp) - exp_term_0 * self.wofz(z2_0, xp=xp))
            output_grid[~ind_pos_y] = xp.conj(-1j * (self.wofz(z1_1, xp=xp) - exp_term_1 * self.wofz(z2_1, xp=xp)))

        return output_grid

    def wofz(self, z, xp=np):
        """
        JAX-compatible Faddeeva function w(z) = exp(-z^2) * erfc(-i z)
        Based on the Poppe–Wijers / Zaghloul–Ali rational approximations.
        Valid for all complex z. JIT + autodiff safe.
        """

        z = xp.asarray(z)
        x = xp.real(z)
        y = xp.imag(z)

        r2 = x * x + y * y
        y2 = y * y
        z2 = z * z
        sqrt_pi = xp.sqrt(xp.pi)

        # --- Regions 1 to 4 ---
        r1_s1 = xp.array([2.5, 2.0, 1.5, 1.0, 0.5])

        t = z
        for coef in r1_s1:
            t = z - coef / t

        w_large = 1j / (t * sqrt_pi)

        # --- Region 5: special small-imaginary case ---
        U5 = xp.array([1.320522, 35.7668, 219.031, 1540.787, 3321.990, 36183.31])
        V5 = xp.array([1.841439, 61.57037, 364.2191, 2186.181,
                       9022.228, 24322.84, 32066.6])

        t = 1 / sqrt_pi
        for u in U5:
            t = u + z2 * t

        s = 1.0
        for v in V5:
            s = v + z2 * s

        w5 = xp.exp(-z2) + 1j * z * t / s

        # --- Region 6: remaining small-|z| region ---
        U6 = xp.array([5.9126262, 30.180142, 93.15558,
                       181.92853, 214.38239, 122.60793])
        V6 = xp.array([10.479857, 53.992907, 170.35400,
                       348.70392, 457.33448, 352.73063, 122.60793])

        t = 1 / sqrt_pi
        for u in U6:
            t = u - 1j * z * t

        s = 1.0
        for v in V6:
            s = v - 1j * z * s

        w6 = t / s

        # --- Regions ---
        reg1 = (r2 >= 62.0) | ((r2 >= 30.0) & (r2 < 62.0) & (y2 >= 1e-13))
        reg2 = ((r2 >= 30) & (r2 < 62) & (y2 < 1e-13)) | ((r2 >= 2.5) & (r2 < 30) & (y2 < 0.072))

        # --- Combine regions using pure array logic ---
        w = w6
        w = xp.where(reg2, w5, w)
        w = xp.where(reg1, w_large, w)

        return w
