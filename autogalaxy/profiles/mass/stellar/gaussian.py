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

        if self.intensity == 0.0:
            return xp.zeros((grid.shape[0], 2))

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
    def deflection_func(u, y, x, npow, axis_ratio, sigma):
        _eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
        )

        return np.exp(-0.5 * np.square(np.divide(_eta_u, sigma))) / (
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
        return np.zeros(shape=grid.shape[0])

    def image_2d_via_radii_from(self, grid_radii: np.ndarray, xp=np):
        """Calculate the intensity of the Gaussian light profile on a grid of radial coordinates.

        Parameters
        ----------
        grid_radii
            The radial distance from the centre of the profile. for each coordinate on the grid.

        Note: sigma is divided by sqrt(q) here.
        """
        return np.multiply(
            self.intensity,
            np.exp(
                -0.5
                * np.square(
                    np.divide(
                        grid_radii.array, self.sigma / np.sqrt(self.axis_ratio(xp))
                    )
                )
            ),
        )

    def axis_ratio(self, xp=np):
        axis_ratio = super().axis_ratio(xp=xp)
        return xp.where(axis_ratio < 0.9999, axis_ratio, 0.9999)

    def zeta_from(self, grid: aa.type.Grid2DLike, xp=np):

        #from scipy.special import wofz

        q = self.axis_ratio(xp)
        q2 = q ** 2.0
        ind_pos_y = grid.array[:, 0] >= 0
        shape_grid = xp.shape(grid)
        output_grid = xp.zeros((shape_grid[0]), dtype=xp.complex128)
        scale_factor = q / (self.sigma * xp.sqrt(2.0 * (1.0 - q2)))

        xs_0 = grid.array[:, 1][ind_pos_y] * scale_factor
        ys_0 = grid.array[:, 0][ind_pos_y] * scale_factor
        xs_1 = grid.array[:, 1][~ind_pos_y] * scale_factor
        ys_1 = -grid.array[:, 0][~ind_pos_y] * scale_factor

        output_grid[ind_pos_y] = -1j * (
            self.wofz(xs_0 + 1j * ys_0, xp=xp)
            - xp.exp(-(xs_0**2.0) * (1.0 - q2) - ys_0 * ys_0 * (1.0 / q2 - 1.0))
            * self.wofz(q * xs_0 + 1j * ys_0 / q, xp=xp)
        )

        output_grid[~ind_pos_y] = xp.conj(
            -1j
            * (
                self.wofz(xs_1 + 1j * ys_1, xp=xp)
                - xp.exp(-(xs_1**2.0) * (1.0 - q2) - ys_1 * ys_1 * (1.0 / q2 - 1.0))
                * self.wofz(q * xs_1 + 1j * ys_1 / q, xp=xp)
            )
        )

        return output_grid


    def wofz(self, z, xp=np):
        """
        JAX-compatible Faddeeva function w(z) = exp(-z^2) * erfc(-i z)
        Based on the Poppe–Wijers / Zaghloul–Ali rational approximations.
        Valid for all complex z. JIT + autodiff safe.
        """

        # y = grid.array[:, 0]
        # x = grid.array[:, 1]
        # z = x + 1j * y

        z = xp.asarray(z, dtype=xp.complex128)
        x = xp.real(z)
        y = xp.imag(z)

        r2 = x * x + y * y
        y2 = y * y
        z2 = z * z
        sqrt_pi = xp.sqrt(xp.pi)

        # --- Region 1: |z|^2 >= 3.8e4 ---
        w1 = 1j / (z * sqrt_pi)

        # --- Region 2: 3.8e4 > |z|^2 >= 256 ---
        w2 = 1j * z / (sqrt_pi * (z2 - 0.5))

        # --- Region 3: 256 > |z|^2 >= 62 ---
        w3 = 1j * (z2 - 1.0) / (z * sqrt_pi * (z2 - 1.5))

        # --- Region 4: 62 > |z|^2 >= 30 and y^2 >= 1e-13 ---
        w4 = 1j * z * (z2 - 2.5) / (sqrt_pi * (z2 * (z2 - 3.0) + 0.75))

        # --- Region 5: special small-imaginary case ---
        U5 = xp.array([1.320522, 35.7668, 219.031, 1540.787, 3321.990, 36183.31], dtype=xp.float64)
        V5 = xp.array([1.841439, 61.57037, 364.2191, 2186.181,
                        9022.228, 24322.84, 32066.6], dtype=xp.float64)

        # Horner form in z^2
        num5 = U5[0]
        for k in range(1, 6):
            num5 = num5 * z2 + U5[k]
        num5 = num5 * z2 + sqrt_pi

        den5 = V5[0]
        for k in range(1, 7):
            den5 = den5 * z2 + V5[k]
        den5 = den5 * z2 + z2

        w5 = xp.exp(-z2) + 1j * z * num5 / den5

        # --- Region 6: remaining small-|z| region ---
        U6 = xp.array([5.9126262, 30.180142, 93.15558,
                        181.92853, 214.38239, 122.60793], dtype=xp.float64)
        V6 = xp.array([10.479857, 53.992907, 170.35400,
                        348.70392, 457.33448, 352.73063, 122.60793], dtype=xp.float64)

        num6 = U6[0]
        for k in range(1, 6):
            num6 = num6 * (-1j * z) + U6[k]
        num6 = num6 * (-1j * z) + sqrt_pi

        den6 = V6[0]
        for k in range(1, 7):
            den6 = den6 * (-1j * z) + V6[k]
        den6 = den6 * (-1j * z) + (-1j * z)

        w6 = num6 / den6

        # --- Combine regions using pure array logic ---
        w = w6
        w = xp.where((r2 >= 2.5) & (y2 < 0.072) & (r2 < 30), w5, w)
        w = xp.where((r2 >= 30) & (r2 < 62) & (y2 < 1e-13), w5, w)
        w = xp.where((r2 >= 30) & (r2 < 62) & (y2 >= 1e-13), w4, w)
        w = xp.where((r2 >= 62) & (r2 < 256), w3, w)
        w = xp.where((r2 >= 256) & (r2 < 3.8e4), w2, w)
        w = xp.where(r2 >= 3.8e4, w1, w)

        return w
