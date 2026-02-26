import numpy as np

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class MGEDecomposer:
    """
    This class speeds up deflection angle calculations of certain mass profiles by decompositing them into many
    Gaussians.

    This follows the method of Shajib 2019 - https://academic.oup.com/mnras/article/488/1/1387/5526256
    """

    def __init__(
        self,
        mass_profile: MassProfile,
    ):
        self.mass_profile = mass_profile


    @property
    def centre(self):
        return self.mass_profile.centre


    @property
    def ell_comps(self):
        return self.mass_profile.ell_comps


    @property
    def transformed_to_reference_frame_grid_from(self):
        return self.mass_profile.transformed_to_reference_frame_grid_from


    @staticmethod
    def kesi(p, xp=np):
        """
        see Eq.(6) of Shajib 2019 1906.08263
        """
        n_list = xp.arange(0, 2 * p + 1, 1)
        return (2.0 * p * xp.log(10) / 3.0 + 2.0 * xp.pi * n_list * 1j) ** (0.5)


    @staticmethod
    def eta(p, xp=np):
        """
        see Eq.(6) of Shajib 2019 1906.00263
        """
        i = xp.arange(1, p, 1)
        kesi_last = 1 / 2 ** p
        k = kesi_last + xp.cumsum(xp.cumprod((p + 1 - i) / i) * kesi_last)

        kesi_list = xp.hstack(
            [xp.array([0.5]), xp.ones(p), k[::-1], xp.array([kesi_last])]
        )
        coef = (-1) ** xp.arange(0, 2 * p + 1, 1)
        eta_const = 2.0 * xp.sqrt(2.0 * xp.pi) * 10 ** (p / 3.0)
        eta_list = coef * eta_const * kesi_list

        return eta_list


    @staticmethod
    def wofz(z, xp=np):
        """
        JAX-compatible Faddeeva function w(z) = exp(-z^2) * erfc(-i z)
        Based on the Poppe–Wijers / Zaghloul–Ali rational approximations.
        Valid for all complex z. JIT + autodiff safe.
        """

        z = xp.asarray(z, dtype=xp.complex128)
        x = xp.real(z)
        y = xp.imag(z)

        r2 = x * x + y * y
        y2 = y * y
        z2 = z * z

        sqrt_pi = xp.asarray(xp.sqrt(xp.pi), dtype=xp.float64)
        inv_sqrt_pi = xp.asarray(1.0 / sqrt_pi, dtype=xp.float64)

        # ---------- Large-|z| continued fraction ----------
        r1_s1 = xp.asarray([2.5, 2.0, 1.5, 1.0, 0.5], dtype=xp.float64)

        t = z
        for c in r1_s1:
            t = z - c / t

        w_large = 1j * inv_sqrt_pi / t

        # ---------- Region 5 ----------
        U5 = xp.asarray(
            [1.320522, 35.7668, 219.031, 1540.787, 3321.990, 36183.31], dtype=xp.float64
        )
        V5 = xp.asarray(
            [1.841439, 61.57037, 364.2191, 2186.181, 9022.228, 24322.84, 32066.6],
            dtype=xp.float64,
        )

        t = inv_sqrt_pi
        for u in U5:
            t = u + z2 * t

        s = xp.asarray(1.0, dtype=xp.float64)
        for v in V5:
            s = v + z2 * s

        w5 = xp.exp(xp.clip(-z2, None, 700.0)) + 1j * z * t / s #clip prevents overflow error

        # ---------- Region 6 ----------
        U6 = xp.asarray(
            [5.9126262, 30.180142, 93.15558, 181.92853, 214.38239, 122.60793],
            dtype=xp.float64,
        )
        V6 = xp.asarray(
            [
                10.479857,
                53.992907,
                170.35400,
                348.70392,
                457.33448,
                352.73063,
                122.60793,
            ],
            dtype=xp.float64,
        )

        t = inv_sqrt_pi
        for u in U6:
            t = u - 1j * z * t

        s = xp.asarray(1.0, dtype=xp.float64)
        for v in V6:
            s = v - 1j * z * s

        w6 = t / s

        # ---------- Region logic ----------
        reg1 = (r2 >= 62.0) | ((r2 >= 30.0) & (r2 < 62.0) & (y2 >= 1e-13))
        reg2 = ((r2 >= 30) & (r2 < 62) & (y2 < 1e-13)) | (
                (r2 >= 2.5) & (r2 < 30) & (y2 < 0.072)
        )

        w = w6
        w = xp.where(reg2, w5, w)
        w = xp.where(reg1, w_large, w)

        return w


    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_2d_via_mge_from(
        self, grid: aa.type.Grid2DLike, xp=np, *, sigma_log_list, func_terms: int = 28, three_D: bool = True, **kwargs,
    ):
        amps, sigmas = self.decompose_convergence_via_mge(
            sigma_log_list=sigma_log_list, func_terms=func_terms, three_D=three_D, xp=xp)

        q = xp.asarray(self.axis_ratio(xp), dtype=xp.float64)

        #sigmas = xp.sqrt(q) * sigma_log_list

        deflection_angles = (
                amps[:, None]
                * sigmas[:, None]
                * xp.sqrt((2.0 * xp.pi) / (1.0 - q**2.0))
                * self.zeta_from(grid=grid, sigma_log_list=sigmas, xp=xp)
        )

        # Add Gaussian profiles
        deflections = xp.sum(deflection_angles, axis=0)

        return self.mass_profile.rotated_grid_from_reference_frame_from(
            xp.vstack((-1.0 * xp.imag(deflections), xp.real(deflections))).T,
            xp=xp,
        )


    def decompose_convergence_via_mge(
        self, sigma_log_list, func_terms: int = 28, three_D: bool = True, xp=np
    ):
        """

        Parameters
        ----------
        func : func
            The function representing the profile that is decomposed into Gaussians.
        normalization
            A normalization factor tyh
        func_terms
            The number of terms used to approximate the input func.
        func_gaussians
            The number of Gaussians used to represent the input func.

        Returns
        -------
        """
        kesis = self.kesi(func_terms, xp=xp)  # kesi in Eq.(6) of 1906.08263
        etas = self.eta(func_terms, xp=xp)  # eta in Eqr.(6) of 1906.08263

        sigmas = xp.asarray(sigma_log_list, dtype=xp.float64) # major axis

        log_sigmas = xp.log(sigmas)
        d_log_sigma = log_sigmas[1] - log_sigmas[0]

        q = xp.asarray(self.axis_ratio(xp), dtype=xp.float64)

        if three_D == True:
            f_sigma = xp.sum(
                etas * xp.real(self.mass_profile.density_3d_func(sigmas.reshape(-1, 1) * kesis, xp=xp)), axis=1
            )
            amplitude_list = f_sigma * d_log_sigma * sigmas
            sigmas = q * sigmas

        else:
            f_sigma = xp.sum(
                etas * xp.real(self.mass_profile.convergence_func(sigmas.reshape(-1, 1) * kesis, xp=xp)), axis=1
            )
            amplitude_list = f_sigma * d_log_sigma / xp.sqrt(2.0 * xp.pi)
            sigmas = xp.sqrt(q) * sigmas

        if xp==np:
            #amplitude_list[0] *= 0.5
            amplitude_list[-1] *= 0.5
        else:
            #amplitude_list = amplitude_list.at[0].multiply(0.5)
            amplitude_list = amplitude_list.at[-1].multiply(0.5)

        return amplitude_list, sigmas


    # def decompose_convergence_sph_via_mge(
    #     self, sigma_log_list, func_terms: int = 28, xp=np
    # ):
    #     """
    #
    #     Parameters
    #     ----------
    #     func : func
    #         The function representing the profile that is decomposed into Gaussians.
    #     normalization
    #         A normalization factor tyh
    #     func_terms
    #         The number of terms used to approximate the input func.
    #     func_gaussians
    #         The number of Gaussians used to represent the input func.
    #
    #     Returns
    #     -------
    #     """
    #     kesis = self.kesi(func_terms, xp=xp)  # kesi in Eq.(6) of 1906.08263
    #     etas = self.eta(func_terms, xp=xp)  # eta in Eqr.(6) of 1906.08263
    #
    #     sigmas = xp.array(sigma_log_list)
    #
    #     #log_sigmas = xp.linspace(xp.log(radii_min), xp.log(radii_max), func_gaussians)
    #     log_sigmas = xp.log(sigmas)
    #     d_log_sigma = log_sigmas[1] - log_sigmas[0]
    #     #sigma_list = xp.exp(log_sigmas)
    #
    #     f_sigma = xp.sum(
    #         etas * xp.real(self.mass_profile.convergence_func(sigmas.reshape(-1, 1) * kesis, xp=xp)), axis=1
    #     )
    #
    #     amplitude_list = f_sigma * d_log_sigma / xp.sqrt(2.0 * xp.pi)
    #     if xp==np:
    #         amplitude_list[0] *= 0.5
    #         amplitude_list[-1] *= 0.5
    #     else:
    #         amplitude_list = amplitude_list.at[0].multiply(0.5)
    #         amplitude_list = amplitude_list.at[-1].multiply(0.5)
    #
    #     return amplitude_list, sigmas


    def axis_ratio(self, xp=np):
        axis_ratio = self.mass_profile.axis_ratio(xp=xp)
        return xp.where(axis_ratio < 0.9999, axis_ratio, 0.9999)


    def zeta_from(self, grid: aa.type.Grid2DLike, sigma_log_list, xp=np):
        q = xp.asarray(self.axis_ratio(xp), dtype=xp.float64)
        q2 = q * q

        y = xp.asarray(grid.array[:, 0], dtype=xp.float64)
        x = xp.asarray(grid.array[:, 1], dtype=xp.float64)

        ind_pos_y = y >= 0

        sigmas = xp.asarray(sigma_log_list, dtype=xp.float64)[:, None]

        scale = q / (
                sigmas * xp.sqrt(xp.asarray(2.0, dtype=xp.float64) * (1.0 - q2))
        )

        xs = x[None, :] * scale
        ys = xp.abs(y)[None, :] * scale

        z1 = xs + 1j * ys
        z2 = q * xs + 1j * ys / q

        exp_term = xp.exp(
            -(xs * xs) * (1.0 - q2)
            - (ys * ys) * (1.0 / q2 - 1.0)
        )

        core = -1j * (
                self.wofz(z1, xp=xp)
                - exp_term * self.wofz(z2, xp=xp)
        )

        return xp.where(ind_pos_y[None, :], core, xp.conj(core))
