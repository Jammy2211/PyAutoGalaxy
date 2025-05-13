from typing import Tuple
import numpy as np

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class dPIE(MassProfile):

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        kappa_scale: float = 0.1,
    ):
        """
        The dual Pseudo-Isothermal mass profile (dPIE) without ellipticity, based on the
        formulation from Eliasdottir (2007): https://arxiv.org/abs/0710.5636.

        This profile describes a circularly symmetric (non-elliptical) projected mass
        distribution with two scale radii (`ra` and `rs`) and a normalization factor
        `kappa_scale`. Although originally called the dPIE (Elliptical), this version
        lacks ellipticity, so the "E" may be a misnomer.

        The projected surface mass density is given by:

        .. math::

            \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                          (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

        (See Eliasdottir 2007, Eq. A3.)

        In this implementation:
        - `ra` and `rs` are scale radii in arcseconds.
        - `kappa_scale` = Σ₀ / Σ_crit is the dimensionless normalization.

        Credit: Jackson O'Donnell for implementing this profile in PyAutoLens.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core scale radius in arcseconds.
        rs
            The outer truncation scale radius in arcseconds.
        kappa_scale
            The dimensionless normalization factor controlling the overall mass.
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

        if ra > rs:
            ra, rs = rs, ra

        self.ra = ra
        self.rs = rs
        self.kappa_scale = kappa_scale

    def _ellip(self):
        ellip = np.sqrt(self.ell_comps[0] ** 2 + self.ell_comps[1] ** 2)
        MAX_ELLIP = 0.99999
        return min(ellip, MAX_ELLIP)

    def _deflection_angle(self, radii):
        """
        For a circularly symmetric dPIE profile, computes the magnitude of the deflection at each radius.
        """
        r_ra = radii / self.ra
        r_rs = radii / self.rs
        # c.f. Eliasdottir '07 eq. A20
        f = r_ra / (1 + np.sqrt(1 + r_ra * r_ra)) - r_rs / (
            1 + np.sqrt(1 + r_rs * r_rs)
        )

        ra, rs = self.ra, self.rs
        # c.f. Eliasdottir '07 eq. A19
        # magnitude of deflection
        alpha = 2 * self.kappa_scale * ra * rs / (rs - ra) * f
        return alpha

    def _convergence(self, radii):
        radsq = radii * radii
        a, s = self.ra, self.rs
        # c.f. Eliasdottir '07 eqn (A3)
        return (
            self.kappa_scale
            * (a * s)
            / (s - a)
            * (1 / np.sqrt(a**2 + radsq) - 1 / np.sqrt(s**2 + radsq))
        )

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        ellip = self._ellip()
        grid_radii = np.sqrt(
            grid[:, 1] ** 2 * (1 - ellip) + grid[:, 0] ** 2 * (1 + ellip)
        )

        # Compute the deflection magnitude of a *non-elliptical* profile
        alpha_circ = self._deflection_angle(grid_radii)

        # This is in axes aligned to the major/minor axis
        deflection_y = alpha_circ * np.sqrt(1 + ellip) * (grid[:, 0] / grid_radii)
        deflection_x = alpha_circ * np.sqrt(1 - ellip) * (grid[:, 1] / grid_radii)

        # And here we convert back to the real axes
        return self.rotated_grid_from_reference_frame_from(
            grid=np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T), **kwargs
        )

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
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
        grid_radii = np.sqrt(
            grid[:, 1] ** 2 * (1 - ellip) + grid[:, 0] ** 2 * (1 + ellip)
        )

        # Compute the convergence and deflection of a *circular* profile
        kappa_circ = self._convergence(grid_radii)
        alpha_circ = self._deflection_angle(grid_radii)

        asymm_term = (
            ellip * (1 - ellip) * grid[:, 1] ** 2
            - ellip * (1 + ellip) * grid[:, 0] ** 2
        ) / grid_radii**2

        # convergence = 1/2 \nabla \alpha = 1/2 \nabla^2 potential
        # The "asymm_term" is asymmetric on x and y, so averages out to
        # zero over all space
        return kappa_circ * (1 - asymm_term) + (alpha_circ / grid_radii) * asymm_term

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        return np.zeros(shape=grid.shape[0])


class dPIESph(dPIE):

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        kappa_scale: float = 0.1,
    ):
        """
        The dual Pseudo-Isothermal mass profile (dPIE) without ellipticity, based on the
        formulation from Eliasdottir (2007): https://arxiv.org/abs/0710.5636.

        This profile describes a circularly symmetric (non-elliptical) projected mass
        distribution with two scale radii (`ra` and `rs`) and a normalization factor
        `kappa_scale`. Although originally called the dPIE (Elliptical), this version
        lacks ellipticity, so the "E" may be a misnomer.

        The projected surface mass density is given by:

        .. math::

            \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                          (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

        (See Eliasdottir 2007, Eq. A3.)

        In this implementation:
        - `ra` and `rs` are scale radii in arcseconds.
        - `kappa_scale` = Σ₀ / Σ_crit is the dimensionless normalization.

        Credit: Jackson O'Donnell for implementing this profile in PyAutoLens.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core scale radius in arcseconds.
        rs
            The outer truncation scale radius in arcseconds.
        kappa_scale
            The dimensionless normalization factor controlling the overall mass.
        """

        # Ensure rs > ra (things will probably break otherwise)
        if ra > rs:
            ra, rs = rs, ra

        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

        self.ra = ra
        self.rs = rs
        self.kappa_scale = kappa_scale

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        radii = self.radial_grid_from(grid=grid, **kwargs)

        alpha = self._deflection_angle(radii)

        # now we decompose the deflection into y/x components
        defl_y = alpha * grid[:, 0] / radii
        defl_x = alpha * grid[:, 1] / radii

        return aa.Grid2DIrregular.from_yx_1d(defl_y, defl_x)

    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
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
        radsq = grid[:, 0] ** 2 + grid[:, 1] ** 2

        return self._convergence(np.sqrt(radsq))

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        return np.zeros(shape=grid.shape[0])
