from typing import Tuple
import numpy as np

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class dPIESph(MassProfile):

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        kappa_scale: float = 0.1,
    ):
        """
        The dual Pseudo-Isothermal Elliptical mass distribution introduced in
        Eliasdottir 2007: https://arxiv.org/abs/0710.5636

        This version is without ellipticity, so perhaps the "E" is a misnomer.

        Corresponds to a projected density profile that looks like:

            \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                          (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

        (c.f. Eliasdottir '07 eqn. A3)

        In this parameterization, ra and rs are the scale radii above in angular
        units (arcsec). The parameter `kappa_scale` is \\Sigma_0 / \\Sigma_crit.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            A scale radius in arc-seconds.
        rs
            The second scale radius in arc-seconds.
        kappa_scale
            Scales the overall normalization of the profile, so related to the mass
        """

        # Ensure rs > ra (things will probably break otherwise)
        if ra > rs:
            ra, rs = rs, ra
        super(MassProfile, self).__init__(centre, (0.0, 0.0))
        self.ra = ra
        self.rs = rs
        self.kappa_scale = kappa_scale

    def _deflection_angle(self, radii):
        '''
        For a circularly symmetric dPIE profile, computes the magnitude of the deflection at each radius.
        '''
        r_ra = radii / self.ra
        r_rs = radii / self.rs
        # c.f. Eliasdottir '07 eq. A20
        f = (
            r_ra / (1 + np.sqrt(1 + r_ra * r_ra))
            - r_rs / (1 + np.sqrt(1 + r_rs * r_rs))
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
            self.kappa_scale * (a * s) / (s - a) *
            (1/np.sqrt(a**2 + radsq) - 1/np.sqrt(s**2 + radsq))
        )

    @aa.grid_dec.to_vector_yx
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        radii = np.sqrt(xoff**2 + yoff**2)

        alpha = self._deflection_angle(radii)

        # now we decompose the deflection into y/x components
        defl_y = alpha * yoff / radii
        defl_x = alpha * xoff / radii
        return aa.Grid2DIrregular.from_yx_1d(
            defl_y, defl_x
        )

    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        # already transformed to center on profile centre so this works
        radsq = (grid[:, 0]**2 + grid[:, 1]**2)
        return self._convergence(np.sqrt(radsq))

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        return np.zeros(shape=grid.shape[0])


class dPIE(dPIESph):
    '''
    The dual Pseudo-Isothermal Elliptical mass distribution introduced in
    Eliasdottir 2007: https://arxiv.org/abs/0710.5636

    Corresponds to a projected density profile that looks like:

        \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                      (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

    (c.f. Eliasdottir '07 eqn. A3)

    In this parameterization, ra and rs are the scale radii above in angular
    units (arcsec). The parameter kappa_scale is \\Sigma_0 / \\Sigma_crit.

    WARNING: This uses the "pseud-elliptical" approximation, where the ellipticity
    is applied to the *potential* rather than the *mass* to ease calculation.
    Use at your own risk! (And TODO Jack: fix this!)
    This approximation is used by the lenstronomy PJAFFE profile (which is the
    same functional form), but not by the lenstool PIEMD (also synonymous with this),
    which correctly solved the differential equations for the mass-based ellipticity.
    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        kappa_scale: float = 0.1,
    ):
        super(MassProfile, self).__init__(centre, ell_comps)
        if ra > rs:
            ra, rs = rs, ra
        self.ra = ra
        self.rs = rs
        self.kappa_scale = kappa_scale

    def _align_to_major_axis(self, ys, xs):
        '''
        Aligns coordinates to the major axis of this halo. Returns (y', x'),
        where x' is along the major axis and y' is along the minor axis.

        Does NOT translate, only rotates.
        '''
        costheta, sintheta = self._cos_and_sin_to_x_axis()
        _xs = (costheta * xs + sintheta * ys)
        _ys = (-sintheta * xs + costheta * ys)
        return _ys, _xs

    def _align_from_major_axis(self, _ys, _xs):
        '''
        Given _ys and _xs as offsets along the minor and major axes,
        respectively, this transforms them back to the regular coordinate
        system.

        Does NOT translate, only rotates.
        '''
        costheta, sintheta = self._cos_and_sin_to_x_axis()
        xs = (costheta * _xs + -sintheta * _ys)
        ys = (sintheta * _xs + costheta * _ys)
        return ys, xs

    def _ellip(self):
        ellip = np.sqrt(self.ell_comps[0]**2 + self.ell_comps[1]**2)
        MAX_ELLIP = 0.99999
        return min(ellip, MAX_ELLIP)

    @aa.grid_dec.to_vector_yx
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)

        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))

        # Compute the deflection magnitude of a *non-elliptical* profile
        alpha_circ = self._deflection_angle(_radii)

        # This is in axes aligned to the major/minor axis
        _defl_xs = alpha_circ * np.sqrt(1 - ellip) * (_xs / _radii)
        _defl_ys = alpha_circ * np.sqrt(1 + ellip) * (_ys / _radii)

        # And here we convert back to the real axes
        defl_ys, defl_xs = self._align_from_major_axis(_defl_ys, _defl_xs)
        return aa.Grid2DIrregular.from_yx_1d(
            defl_ys, defl_xs
        )

    @aa.grid_dec.to_array
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)

        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))

        # Compute the convergence and deflection of a *circular* profile
        kappa_circ = self._convergence(_radii)
        alpha_circ = self._deflection_angle(_radii)

        asymm_term = (ellip * (1 - ellip) * _xs**2 - ellip * (1 + ellip) * _ys**2) / _radii**2

        # convergence = 1/2 \nabla \alpha = 1/2 \nabla^2 potential
        # The "asymm_term" is asymmetric on x and y, so averages out to
        # zero over all space
        return kappa_circ * (1 - asymm_term) + (alpha_circ / _radii) * asymm_term

    @aa.grid_dec.to_array
    def potential_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        return np.zeros(shape=grid.shape[0])