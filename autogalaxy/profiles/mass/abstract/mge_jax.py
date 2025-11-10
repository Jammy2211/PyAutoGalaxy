# from .jax_utils import w_f_approx


class MassProfileMGE:
    """
    This class speeds up deflection angle calculations of certain mass profiles by decompositing them into many
    Gaussians.

    This follows the method of Shajib 2019 - https://academic.oup.com/mnras/article/488/1/1387/5526256
    """

    def __init__(self):
        pass

    @staticmethod
    def zeta_from(grid, amps, sigmas, axis_ratio):
        """
        The key part to compute the deflection angle of each Gaussian.
        """
        q2 = axis_ratio**2.0

        scale_factor = axis_ratio / xp.sqrt(2.0 * (1.0 - q2))

        xs = xp.array((grid.array[:, 1] * scale_factor).copy())
        ys = xp.array((grid.array[:, 0] * scale_factor).copy())

        y_sign = xp.sign(ys)
        ys = ys * y_sign

        z = xs + 1j * ys
        zq = axis_ratio * xs + 1j * ys / axis_ratio
        expv = -(xs**2.0) * (1.0 - q2) - ys**2.0 * (1.0 / q2 - 1.0)
        sigma_ = sigmas.reshape((-1,) + (1,) * xs.ndim)
        inv_sigma_ = 1 / sigma_
        amps_ = amps.reshape((-1,) + (1,) * xs.ndim)

        # process as one big vectorized calculation
        # could try `jax.lax.scan` instead if this is too much memory
        w = w_f_approx(inv_sigma_ * z)
        wq = w_f_approx(inv_sigma_ * zq)
        exp_factor = xp.exp(inv_sigma_**2 * expv)

        sigma_func_real = w.imag - exp_factor * wq.imag
        sigma_func_imag = (-w.real + exp_factor * wq.real) * y_sign

        output_grid = sigma_ * amps_ * (sigma_func_real + 1j * sigma_func_imag)
        return output_grid.sum(axis=0)

    @staticmethod
    def kesi(p):
        """
        see Eq.(6) of 1906.08263
        """
        n_list = xp.arange(0, 2 * p + 1, 1)
        return (2.0 * p * xp.log(10) / 3.0 + 2.0 * xp.pi * n_list * 1j) ** (0.5)

    @staticmethod
    def eta(p):
        """
        see Eq.(6) of 1906.00263
        """

        i = xp.arange(1, p, 1)
        kesi_last = 1 / 2**p
        k = kesi_last + xp.cumsum(xp.cumprod((p + 1 - i) / i) * kesi_last)

        kesi_list = xp.hstack(
            [xp.array([0.5]), xp.ones(p), k[::-1], xp.array([kesi_last])]
        )
        coef = (-1) ** xp.arange(0, 2 * p + 1, 1)
        eta_const = 2.0 * xp.sqrt(2.0 * xp.pi) * 10 ** (p / 3.0)
        eta_list = coef * kesi_list
        return eta_const, eta_list

    def decompose_convergence_via_mge(self):
        raise NotImplementedError()

    def _decompose_convergence_via_mge(
        self, func, radii_min, radii_max, func_terms=28, func_gaussians=20
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

        kesis = self.kesi(func_terms)  # kesi in Eq.(6) of 1906.08263
        eta_constant, eta_n = self.eta(func_terms)  # eta in Eqr.(6) of 1906.08263

        # sigma is sampled from logspace between these radii.

        log_sigmas = xp.linspace(xp.log(radii_min), xp.log(radii_max), func_gaussians)
        d_log_sigma = log_sigmas[1] - log_sigmas[0]
        sigma_list = xp.exp(log_sigmas)

        f_sigma = eta_constant * xp.sum(
            eta_n * xp.real(func(sigma_list.reshape(-1, 1) * kesis)), axis=1
        )
        amplitude_list = f_sigma * d_log_sigma / xp.sqrt(2.0 * xp.pi)
        amplitude_list = amplitude_list.at[0].multiply(0.5)
        amplitude_list = amplitude_list.at[-1].multiply(0.5)

        return amplitude_list, sigma_list

    def convergence_2d_via_mge_from(self, grid_radii):
        raise NotImplementedError()

    def _convergence_2d_via_mge_from(
        self, grid_radii, func_terms=28, func_gaussians=20
    ):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        amps, sigmas = self.decompose_convergence_via_mge(
            func_terms=func_terms, func_gaussians=func_gaussians
        )

        inv_sigma_ = 1 / sigmas.reshape((-1,) + (1,) * grid_radii.array.ndim)
        amps_ = amps.reshape((-1,) + (1,) * grid_radii.array.ndim)
        convergence = amps_ * xp.exp(-0.5 * (grid_radii.array * inv_sigma_) ** 2)
        return convergence.sum(axis=0)

    def _deflections_2d_via_mge_from(
        self, grid, sigmas_factor=1.0, func_terms=28, func_gaussians=20
    ):
        axis_ratio = xp.min(xp.array([self.axis_ratio(xp), 0.9999]))

        amps, sigmas = self.decompose_convergence_via_mge(
            func_terms=func_terms, func_gaussians=func_gaussians
        )
        sigmas *= sigmas_factor

        angle = self.zeta_from(
            grid=grid, amps=amps, sigmas=sigmas, axis_ratio=axis_ratio
        )

        angle *= xp.sqrt((2.0 * xp.pi) / (1.0 - axis_ratio**2.0))

        return self.rotated_grid_from_reference_frame_from(
            xp.vstack((-angle.imag, angle.real)).T
        )
