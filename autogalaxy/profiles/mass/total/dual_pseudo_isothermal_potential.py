from typing import Tuple
import jax.numpy as jnp

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class dPIEP(MassProfile):

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        b0: float = 1.0,
    ):
        """
        The dual Pseudo Isothermal Elliptical Potential (dPIEP) with pseudo-ellipticity on potential, based on the
        formulation from Eliasdottir (2007): https://arxiv.org/abs/0710.5636.

        This profile describes a circularly symmetric (non-elliptical) projected mass
        distribution with two scale radii (`ra` and `rs`) and a normalization factor
        `kappa_scale`. Although originally called the dPIEP (Elliptical), this version
        lacks ellipticity, so the "E" may be a misnomer.

        The projected surface mass density is given by:

        .. math::

            \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                          (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

        (See Eliasdottir 2007, Eq. A3.)

        In this implementation:
        - `ra` is the core radius in unit of arcseconds.
        - `b0` is the lens strength in unit of arcseconds, when ra->0 & rs->\\infty & q->1, b0 is the Einstein radius.
          `b0` is related to the central velocity dispersion \\sigma_0: b_0 = 4\\pi * \\sigma_0^2 / c^2 * (D_{LS} / D_{S})
          `b0` is not in the Intermediate-Axis-Convention for its r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2

        Credit: Jackson O'Donnell for implementing this profile in PyAutoLens.
        Note: To ensure consistency, kappa_scale was replaced with b0, and the corresponding code was adjusted accordingly.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core scale radius in arcseconds.
        rs
            The outer truncation scale radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

        if ra > rs:
            ra, rs = rs, ra

        self.ra = ra
        self.rs = rs
        self.b0 = b0

    def _ellip(self):
        ellip = jnp.sqrt(self.ell_comps[0] ** 2 + self.ell_comps[1] ** 2)
        MAX_ELLIP = 0.99999
        return min(ellip, MAX_ELLIP)

    def _deflection_angle(self, radii):
        """
        For a circularly symmetric dPIEP profile, computes the magnitude of the deflection at each radius.
        """
        a, s = self.ra, self.rs
        radii = jnp.maximum(radii, 1e-8)
        f = radii / (a + jnp.sqrt(a**2 + radii**2)) - radii / (
            s + jnp.sqrt(s**2 + radii**2)
        )

        # c.f. Eliasdottir '07 eq. A23
        # magnitude of deflection
        # alpha = self.E0 * (s + a) / s * f
        alpha = self.b0 * s / (s - a) * f
        return alpha

    def _convergence(self, radii):

        radsq = radii * radii
        a, s = self.ra, self.rs

        return (
            self.b0
            / 2
            * s
            / (s - a)
            * (1 / jnp.sqrt(a**2 + radsq) - 1 / jnp.sqrt(s**2 + radsq))
        )

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        ellip = self._ellip()
        grid_radii = jnp.sqrt(
            grid.array[:, 1] ** 2 * (1 - ellip) + grid.array[:, 0] ** 2 * (1 + ellip)
        )

        # Compute the deflection magnitude of a *non-elliptical* profile
        alpha_circ = self._deflection_angle(grid_radii)

        # This is in axes aligned to the major/minor axis
        deflection_y = (
            alpha_circ * jnp.sqrt(1 + ellip) * (grid.array[:, 0] / grid_radii)
        )
        deflection_x = (
            alpha_circ * jnp.sqrt(1 - ellip) * (grid.array[:, 1] / grid_radii)
        )

        # And here we convert back to the real axes
        return self.rotated_grid_from_reference_frame_from(
            grid=jnp.multiply(1.0, jnp.vstack((deflection_y, deflection_x)).T), **kwargs
        )

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See
        *aa.grid_2d_to_structure* for a description of the output.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        ellip = self._ellip()
        grid_radii = jnp.sqrt(
            grid.array[:, 1] ** 2 * (1 - ellip) + grid.array[:, 0] ** 2 * (1 + ellip)
        )

        # Compute the convergence and deflection of a *circular* profile
        kappa_circ = self._convergence(grid_radii)
        alpha_circ = self._deflection_angle(grid_radii)

        asymm_term = (
            ellip * (1 - ellip) * grid.array[:, 1] ** 2
            - ellip * (1 + ellip) * grid.array[:, 0] ** 2
        ) / grid_radii**2

        # convergence = 1/2 \nabla \alpha = 1/2 \nabla^2 potential
        # The "asymm_term" is asymmetric on x and y, so averages out to
        # zero over all space
        return kappa_circ * (1 - asymm_term) + (alpha_circ / grid_radii) * asymm_term

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        return jnp.zeros(shape=grid.shape[0])


class dPIEPSph(dPIEP):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        b0: float = 1.0,
    ):
        """
        The dual Pseudo-Isothermal mass profile (dPIEP) without ellipticity, based on the
        formulation from Eliasdottir (2007): https://arxiv.org/abs/0710.5636.

        This profile describes a circularly symmetric (non-elliptical) projected mass
        distribution with two scale radii (`ra` and `rs`) and a normalization factor
        `kappa_scale`. Although originally called the dPIEP (Elliptical), this version
        lacks ellipticity, so the "E" may be a misnomer.

        The projected surface mass density is given by:

        .. math::

            \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                          (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

        (See Eliasdottir 2007, Eq. A3.)

        In this implementation:
        - `ra` is the core radius in unit of arcseconds.
        - `b0` is the lens strength in unit of arcseconds, when ra->0 & rs->\\infty & q->1, b0 is the Einstein radius.
          `b0` is related to the central velocity dispersion \\sigma_0: b_0 = 4\\pi * \\sigma_0^2 / c^2 * (D_{LS} / D_{S})
          `b0` is not in the Intermediate-Axis-Convention for its r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2

        Credit: Jackson O'Donnell for implementing this profile in PyAutoLens.
        Note: This dPIEPSph should be the same with dPIEMDSph for their same mathamatical formulations.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core scale radius in arcseconds.
        rs
            The outer truncation scale radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """

        # Ensure rs > ra (things will probably break otherwise)
        if ra > rs:
            ra, rs = rs, ra

        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.ra = ra
        self.rs = rs
        self.b0 = b0

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        radii = self.radial_grid_from(grid=grid, **kwargs)

        alpha = self._deflection_angle(radii.array)

        # now we decompose the deflection into y/x components
        defl_y = alpha * grid.array[:, 0] / radii.array
        defl_x = alpha * grid.array[:, 1] / radii.array

        return aa.Grid2DIrregular.from_yx_1d(defl_y, defl_x)

    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See
        *aa.grid_2d_to_structure* for a description of the output.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        # already transformed to center on profile centre so this works
        radsq = grid.array[:, 0] ** 2 + grid.array[:, 1] ** 2

        return self._convergence(jnp.sqrt(radsq))

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        return jnp.zeros(shape=grid.shape[0])
