from typing import Tuple
import numpy as np

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

# Within this profile family, PIEMD, dPIEMD, and dPIEMDSph are directly ported from Lenstool's C code, and have been thoroughly annotated and adapted for PyAutoLens.
# The dPIEP and dPIEPSph profiles are modified from the original `dPIE` and `dPIESph`, which were implemented to PyAutoLens by Jackson O'Donnell.

def _ci05(x, y, eps, rcore):
    """
    Returns the first derivatives of the lens potential as complex number I'* = (∂ψ/∂x + i ∂ψ/∂y) / E0 for PIEMD at any positions (x,y), 
    see Kassiola & Kovner(1993) Eq. 4.1.2, which is the integral of Eq. 2.3.8.
    Note here b0(or called E0) is out of the `_ci05`.

    Parameters
    ----------
    eps
        The ellipticity of the corresponding profiles.
    rcore
        The inner core radius.
    Returns
    -------
    complex
        The value of the I'* term.
    """
    if eps < 1e-10:
        eps = 1e-10
    sqe = np.sqrt(eps)
    axis_ratio = (1.0 - eps) / (1.0 + eps)
    cxro = (1.0 + eps) * (1.0 + eps)
    cyro = (1.0 - eps) * (1.0 - eps)
    rem2 = x * x / cxro + y * y / cyro
    ##### I'* = zres = zci * ln(zis) = zci * ln(znum / zden), see Eq. 4.1.2 #####

    # Define intermediate complex variables
    zci = np.complex128(complex(0.0, -0.5 * (1.0 - eps * eps) / sqe))
    znum = np.complex128(axis_ratio * x + 1j * (2.0 * sqe * np.sqrt(rcore * rcore + rem2) - y / axis_ratio))
    zden = np.complex128(x + 1j * (2.0 * rcore * sqe - y))
        
    # zis = znum / zden = (a+bi)/(c+di) = [(ac+bd)+(bc-ad i)] / (c^2+d^2)
    norm = zden.real * zden.real + zden.imag * zden.imag  # |zden|^2
    zis_re = (znum.real * zden.real + znum.imag * zden.imag) / norm
    zis_im = (znum.imag * zden.real - znum.real * zden.imag) / norm
    zis = np.complex128(zis_re + 1j * zis_im)
    
    # ln(zis) = ln(|zis|) + i*Arg(zis)
    zis_mag = np.abs(zis)
    zis_re = np.log(zis_mag)
    zis_im = np.angle(zis)
    zis = np.complex128(zis_re + 1j * zis_im)
    
    # I'* = zres = zci * ln(zis)
    zres = zci * zis
    
    return zres

def _ci05f(x, y, eps, rcore, rcut):
    """
    Returns the first derivatives of the lens potential as complex number I'* = (∂ψ/∂x + i ∂ψ/∂y) / (b0 * ra / (rs - ra)) for dPIEMD at any positions (x,y), 
    which is the integral of Eq. 2.3.8 in  Kassiola & Kovner(1993). 
    
    Note here (b0 * ra / (rs - ra)) is out of the `_ci05f`. The only difference of integral of Eq. 2.3.8 between dPIEMD and PIEMD is the \\kappa:
    \\kappa(r_{em})_{dPIEMD} = rs / (rs - ra) * (\\kappa_{PIEMD,ra} - \\kappa_{PIEMD,rs}).
    I*_{dPIEMD} = ra / (rs - ra) * (I*_{PIEMD}(ra) - I*_{PIEMD}(ra))

    Parameters
    ----------
    eps
        The ellipticity of the corresponding profiles.
    rcore
        The inner core radius.
    rcut
        The outer cut radius.
    Returns
    -------
    complex
        The value of the I'* term.
    """
    if eps < 1e-10:
        eps = 1e-10
    sqe = np.sqrt(eps)
    axis_ratio = (1.0 - eps) / (1.0 + eps)
    cxro = (1.0 + eps) * (1.0 + eps)
    cyro = (1.0 - eps) * (1.0 - eps)
    rem2 = x * x / cxro + y * y / cyro

    ##### I'* = zres_rc - zres_rcut = zci * ln(zis_rc / zis_rcut) = zci * ln((znum_rc / zden_rc) / (znum_rcut / zden_rcut)) #####

    # Define intermediate complex variables
    zci = np.complex128(complex(0.0, -0.5 * (1.0 - eps * eps) / sqe))
    znum_rc = np.complex128(axis_ratio * x + 1j * (2.0 * sqe * np.sqrt(rcore * rcore + rem2) - y / axis_ratio)) # a + bi
    zden_rc = np.complex128(x + 1j * (2.0 * rcore * sqe - y)) # c + di
    znum_rcut = np.complex128(axis_ratio * x + 1j * (2.0 * sqe * np.sqrt(rcut * rcut + rem2) - y / axis_ratio)) # a + ei
    zden_rcut = np.complex128(x + 1j * (2.0 * rcut * sqe - y)) # c + fi

    # zis_rc = znum_rc / zden_rc = (a+bi)/(c+di)
    # zis_rcut = znum_rcut / zden_rcut = (a+ei)/(c+fi)
    # zis_tot = zis_rc / zis_rcut = (znum_rc / zden_rc) / (znum_rcut / zden_rcut)
    #                             = [(ac - bf) + (af + bc)i] / [(ac - de) + (ad + ce)i]
    #                             = (aa + bb*i) / (cc + dd*i)
    #                             = (aa + bb*i) * (cc -dd*i) / (cc^2 + dd^2)
    #                             = [(aa*cc + bb*dd) / (cc^ + dd^2)] + [(bb*cc - aa*dd) / (cc^2 + dd^2)]*i
    #                             =                 aaa              +                 bbb*i
    aa = znum_rc.real * zden_rc.real - znum_rc.imag * zden_rcut.imag # ac - bf
    bb = znum_rc.real * zden_rcut.imag + znum_rc.imag * zden_rc.real # af + bc
    cc = znum_rc.real * zden_rc.real - zden_rc.imag * znum_rcut.imag # ac - de
    dd = znum_rc.real * zden_rc.imag + zden_rc.real * znum_rcut.imag # ad + ce
    norm = cc * cc + dd * dd
    aaa = (aa * cc + bb * dd) / norm
    bbb = (bb * cc - aa * dd) / norm
    zis_tot = np.complex128(aaa + 1j * bbb)

    # ln(zis_tot) = ln(|zis_tot|) + i*Arg(zis_tot)
    zis_tot_mag = np.abs(zis_tot)
    zr_re = np.log(zis_tot_mag)
    zr_im = np.angle(zis_tot)
    zr = np.complex128(zr_re + 1j * zr_im)

    # I'* = zci * ln(zis_tot)
    zres = zci * zr

    return zres

def _mdci05(x, y, eps, rcore, b0):
    """
    Returns the second derivatives (Hessian matrix) of the lens potential as complex number for PIEMD at any positions (x,y):
    ∂²ψ/∂x² = Re(∂I*/∂x), ∂²ψ/∂y² = Im(∂I*/∂y), ∂²ψ/∂x∂y = ∂²ψ/∂y∂x = Im(∂I*/∂x) = Re(∂I*/∂y)
    see Kassiola & Kovner(1993) Eq. 4.1.4.

    Parameters
    ----------
    eps
        The ellipticity of the corresponding profiles.
    rcore
        The inner core radius.
    Returns
    -------
    complex
        The value of the I'* term.
    """
    if eps < 1e-10:
        eps = 1e-10

    # Calculate intermediate values
    # I*(x,y) = b0 * ci * (-i) * (ln{ q * x + (2.0 * sqe * wrem - y * 1/q )*i} - ln{ x + (2.0 * rcore * sqe - y)*i})
    #         = b0 * ci * (-i) * (ln{ q * x + num1*i} - ln{ x + num2*i})
    #         = b0 * ci * (-i) * (ln{u(x,y)} - ln{v(x,y)})
    sqe = np.sqrt(eps)
    axis_ratio = (1.0 - eps) / (1.0 + eps)
    axis_ratio_inv = 1.0 / axis_ratio
    cxro = (1.0 + eps) * (1.0 + eps)
    cyro = (1.0 - eps) * (1.0 - eps)
    ci = 0.5 * (1.0 - eps * eps) / sqe
    wrem = np.sqrt(rcore * rcore + x * x / cxro + y * y / cyro) # √(w(x,y))
    num1 = 2.0 * sqe * wrem - y * axis_ratio_inv
    den1 = axis_ratio * axis_ratio * x * x + num1 * num1 # |q * x + num1*i|^2
    num2 = 2.0 * rcore * sqe - y
    den2 = x * x + num2 * num2 # |x + num2*i|^2

    # eg. 
    # ∂²ψ/∂x² = Re(∂I*/∂x) = b0 * didxre
    # ∂I*/∂x = b0 * ci * (-i) * ∂(ln{u(x,y)} - ln{v(x,y)})∂x
    #        = b0 * ci * (-i) * (1/u * ∂u/∂x - 1/v * ∂v/∂x)
    # ∂u/∂x = q + ∂(num1)/∂x * i
    #       = q + [2.0 * sqe * ∂(wrem)/∂x] * i
    #       = q + [2.0 * sqe * ∂(√(w(x,y)))/∂x] * i
    #       = q + [2.0 * sqe * x / cxro / wrem] * i
    # 1/u * ∂u/∂x = {q + [2.0 * sqe * x / cxro / wrem] * i}  /  {q * x + num1*i}
    #             = {q + [2.0 * sqe * x / cxro / wrem] * i} * {q * x - num1*i}  /  |q * x + num1*i|^2
    #             = {q + [2.0 * sqe * x / cxro / wrem] * i} * {q * x - (2.0 * sqe * wrem - y / q)*i}  /  den1
    #             = {q^2 * x + 4.0 * sqe^2 * x - y / q * 2.0 * sqe * x / cxro / wrem} / den1 + q * { (2.0 * sqe * x^2 / cxro / wrem) - (2.0 * sqe * wrem - y / q)} / den1 * i
    #             = {x - 2.0 * sqe * x * y * q / cyro / wrem} / den1 + q * { (2.0 * sqe * x^2 / cxro / wrem) - (2.0 * sqe * wrem - y / q)} / den1 * i
    # (-i) * (1/u * ∂u/∂x) = (2.0 * sqe * x * y * q / cyro / wrem - x) / den1 * i 
    #                      + q * { (2.0 * sqe * x^2 / cxro / wrem) - (2.0 * sqe * wrem - y / q)} / den1
    # ∂v/∂x = 1 + ∂(num2)/∂x * i
    #       = 1
    # 1/v * ∂v/∂x = 1 / (x + num2*i)
    #             = (x - num2*i) / |x + num2*i|^2
    #             = (x - num2*i) / den2
    # -(-i) * (1/v * ∂v/∂x) = (x*i + num2) / den2

    # ∂I*/∂x = b0 * ci * {(-i) * (1/u * ∂u/∂x) - (-i) * (1/v * ∂v/∂x)}

    # Compute second derivatives
    didxre = ci * (
        axis_ratio * (2.0 * sqe * x * x / cxro / wrem - 2.0 * sqe * wrem + y * axis_ratio_inv) / den1
        + num2 / den2
    )
    didyre = ci * (
        (2.0 * sqe * x * y * axis_ratio / cyro / wrem - x) / den1
        + x / den2
    )
    didyim = ci * (
        (2.0 * sqe * wrem * axis_ratio_inv - y * axis_ratio_inv * axis_ratio_inv - 4.0 * eps * y / cyro
        + 2.0 * sqe * y * y / cyro / wrem * axis_ratio_inv) / den1
        - num2 / den2
    )

    # Construct Hessian matrix components
    a = b0 * didxre  # ∂²ψ/∂x²
    b = b0 * didyre  # ∂²ψ/∂x∂y
    c = b0 * didyre  # ∂²ψ/∂y∂x
    d = b0 * didyim  # ∂²ψ/∂y²

    return a,b,c,d

class PIEMD(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        b0: float = 0.1,
    ):
        """
        The Pseudo Isothermal Elliptical Mass Distribution(PIEMD) profiles, based on the formulaiton from 
        Kassiola & Kovner(1993) https://articles.adsabs.harvard.edu/pdf/1993ApJ...417..450K.
        This profile is ported from Lenstool's C code, which has the same formulation.

        This proflie describes an elliptic isothermal mass distribution with a finite core: 
        \\rho \\propto (ra^2 + R^2)^{-1}

        The convergence is given by:
        \\kappa(r_{em}) = \\kappa_0 * ra / \\sqrt{ ra^2 + r_{em}^2 }
        (see Kassiola & Kovner(1993), Eq. 4.1.1)
        where r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2, (see Kassiola & Kovner(1993), Eq. 2.3.6)
        and \\kappa_0 = b_0 / 2 / r_a. 

        In this implementation:
        - `ra` is the core radius in unit of arcseconds.
        - `b0` is the lens strength in unit of arcseconds, when ra->0 & q->1, b0 is the Einstein radius. 
          `b0` is related to the central velocity dispersion \\sigma_0: b_0 = 4\\pi * \\sigma_0^2 / c^2 * (D_{LS} / D_{S}).
          `b0` is not in the Intermediate-Axis-Convention for its r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

        self.ra = ra
        self.b0 = b0
    
    def _ellip(self):
        ellip = np.sqrt(self.ell_comps[0] ** 2 + self.ell_comps[1] ** 2)
        MAX_ELLIP = 0.99999
        return min(ellip, MAX_ELLIP)
    
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
        factor = self.b0
        zis = _ci05(x=grid[:, 1], y=grid[:, 0], eps=ellip, rcore=self.ra)

        # This is in axes aligned to the major/minor axis
        deflection_x = zis.real
        deflection_y = zis.imag

        # And here we convert back to the real axes
        return self.rotated_grid_from_reference_frame_from(
            grid=np.multiply(factor, np.vstack((deflection_y, deflection_x)).T), **kwargs
        )

    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def analytical_hessian_2d_from(self, grid: 'aa.type.Grid2DLike', **kwargs):
        """
        Calculate the hessian matrix on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        grid = np.asarray(grid)
        if grid.ndim != 2 or grid.shape[1] != 2:
            raise ValueError("Grid must be a 2D array with shape (n, 2)")
        ellip = self._ellip()

        hessian_xx, hessian_xy, hessian_yx, hessian_yy = _mdci05(
            x=grid[:, 1], y=grid[:, 0], eps=ellip, rcore=self.ra, b0=self.b0
        )

        return hessian_yy, hessian_xy, hessian_yx, hessian_xx
    
    def analytical_magnification_2d_from(self, grid: 'aa.type.Grid2DLike', **kwargs):

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.analytical_hessian_2d_from(
            grid=grid
        )

        det_A = (1 - hessian_xx) * (1 - hessian_yy) - hessian_xy * hessian_yx

        return aa.Array2D(values=1.0 / det_A, mask=grid.mask)
    
class dPIEMD(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.0,
        rs: float = 2.0,
        b0: float = 0.1,
    ):
        """
        The dual Pseudo Isothermal Elliptical Mass Distribution(dPIEMD) profiles, which is a *two component PIEMD* with both a core radius and a truncation radius, 
        see Eliasdottir (2007): https://arxiv.org/abs/0710.5636
        This profile is ported from Lenstool's C code, which has the same formulation.

        This proflie describes an elliptic isothermal mass distribution with a finite core, \\rho \~ r^{-2} while in the transition region (ra<=R<=rs), 
        and \\rho \~ r^{-4} in the outer parts: 
        \\rho \\propto [(ra^2 + R^2) (rs^2 + R^2)]^{-1}

        The convergence is given by two PIEMD with core radius ra and rs: 
        \\kappa(r_{em}) = rs / (rs - ra) * (\\kappa_{PIEMD,ra} - \\kappa_{PIEMD,rs})
                        = b_0 / 2 * rs / (rs - ra) * ( \\frac{1}{\\sqrt{ ra^2 + r_{em}^2}} - \\frac{1}{\\sqrt{ rs^2 + r_{em}^2}} )
        where r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2. 
        Note in Eliasdottir (2007), E0 = 6\\pi * \\sigma_{dPIE}^2 / c^2 * (D_{LS} / D_{S}). Eliasdottir's E0 is not the same as E0 in Kassiola & Kovner(1993) which is also b0.
        There is \\frac{\\sigma_{dPIE}^2}{\\sigma_0^2} = \\frac{2}{3} \frac{rs^2}{rs^2-ra^2}, 
        thus E0(Kassiola & Kovner(1993)) = b0 = E0(Eliasdottir (2007)) * (rs^2 - ra^2) / rs^2. So when s->\\infty and a->0, they are equivalent.

        In this implementation:
        - `ra` is the core radius in unit of arcseconds.
        - `rs` is the truncation radius in unit of arcseconds.
        - `b0` is the lens strength in unit of arcseconds, when ra->0 & rs->\\infty & q->1, b0 is the Einstein radius. 
          `b0` is related to the central velocity dispersion \\sigma_0: b_0 = 4\\pi * \\sigma_0^2 / c^2 * (D_{LS} / D_{S})
          `b0` is not in the Intermediate-Axis-Convention for its r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core radius in arcseconds.
        rs
            The outer truncation radius in arcseconds.
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
        ellip = np.sqrt(self.ell_comps[0] ** 2 + self.ell_comps[1] ** 2)
        MAX_ELLIP = 0.99999
        return min(ellip, MAX_ELLIP)
    
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
        factor = self.b0 * self.rs / (self.rs - self.ra)
        zis = _ci05f(
            x=grid[:, 1], y=grid[:, 0], eps=ellip, rcore=self.ra, rcut=self.rs
        )

        # This is in axes aligned to the major/minor axis
        deflection_x = zis.real
        deflection_y = zis.imag

        # And here we convert back to the real axes
        return self.rotated_grid_from_reference_frame_from(
            grid=np.multiply(factor, np.vstack((deflection_y, deflection_x)).T), **kwargs
        )
    
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def analytical_hessian_2d_from(self, grid: 'aa.type.Grid2DLike', **kwargs):
        """
        Calculate the hessian matrix on a grid of (y,x) arc-second coordinates.
        Hessian_dPIEMD = rs * (rs - ra) * ( Hessian_PIEMD(ra) - Hessian_PIEMD(rs))

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        grid = np.asarray(grid)
        if grid.ndim != 2 or grid.shape[1] != 2:
            raise ValueError("Grid must be a 2D array with shape (n, 2)")
        ellip = self._ellip()
        
        t05 = self.rs / (self.rs - self.ra)
        g05c_a, g05c_b, g05c_c, g05c_d = _mdci05(
            x=grid[:, 1], y=grid[:, 0], eps=ellip, rcore=self.ra, b0=self.b0
        )
        g05cut_a, g05cut_b, g05cut_c, g05cut_d = _mdci05(
            x=grid[:, 1], y=grid[:, 0], eps=ellip, rcore=self.rs, b0=self.b0
        )

        # Compute Hessian matrix components
        hessian_xx = t05 * (g05c_a - g05cut_a)
        hessian_xy = t05 * (g05c_b - g05cut_b)
        hessian_yx = t05 * (g05c_c - g05cut_c)
        hessian_yy = t05 * (g05c_d - g05cut_d)

        return hessian_yy, hessian_xy, hessian_yx, hessian_xx
    
    def analytical_magnification_2d_from(self, grid: 'aa.type.Grid2DLike', **kwargs):
        
        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.analytical_hessian_2d_from(
            grid=grid
        )

        det_A = (1 - hessian_xx) * (1 - hessian_yy) - hessian_xy * hessian_yx

        return aa.Array2D(values=1.0 / det_A, mask=grid.mask)
    
class dPIEMDSph(dPIEMD):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        b0: float = 1.0
    ):
        """
        The dual Pseudo Isothermal Elliptical Mass Distribution(dPIEMD) profiles without ellipticity, which is a *two component PIEMD* with both a core radius and a truncation radius, 
        see Eliasdottir (2007): https://arxiv.org/abs/0710.5636
        This profile is ported from Lenstool's C code, which has the same formulation.

        This proflie describes an spherical isothermal mass distribution with a finite core, \\rho \~ r^{-2} while in the transition region (ra<=R<=rs), 
        and \\rho \~ r^{-4} in the outer parts: 
        \\rho \\propto [(ra^2 + R^2) (rs^2 + R^2)]^{-1}

        The convergence is given by two PIEMD with core radius ra and rs: 
        \\kappa(r_{em}) = rs / (rs - ra) * (\\kappa_{PIEMD,ra} - \\kappa_{PIEMD,rs})
                        = b_0 / 2 * rs / (rs - ra) * ( \\frac{1}{\\sqrt{ ra^2 + r_{em}^2}} - \\frac{1}{\\sqrt{ rs^2 + r_{em}^2}} )
        where r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2. 
        Note in Eliasdottir (2007), E0 = 6\\pi * \\sigma_{dPIE}^2 / c^2 * (D_{LS} / D_{S}). Eliasdottir's E0 is not the same as E0 in Kassiola & Kovner(1993) which is also b0.
        There is \\frac{\\sigma_{dPIE}^2}{\\sigma_0^2} = \\frac{2}{3} \frac{rs^2}{rs^2-ra^2}, 
        thus E0(Kassiola & Kovner(1993)) = b0 = E0(Eliasdottir (2007)) * (rs^2 - ra^2) / rs^2. So when s->\\infty and a->0, they are equivalent.

        In this implementation:
        - `ra` is the core radius in unit of arcseconds.
        - `rs` is the truncation radius in unit of arcseconds.
        - `b0` is the lens strength in unit of arcseconds, when ra->0 & rs->\\infty & q->1, b0 is the Einstein radius. 
          `b0` is related to the central velocity dispersion \\sigma_0: b_0 = 4\\pi * \\sigma_0^2 / c^2 * (D_{LS} / D_{S})
          `b0` is not in the Intermediate-Axis-Convention for its r_{em}^2 = x^2 / (1 + \\epsilon)^2 + y^2 / (1 - \\epsilon)^2

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ra
            The inner core radius in arcseconds.
        rs
            The outer truncation radius in arcseconds.
        b0
            The lens strength in arcseconds.
        """
        super().__init__(centre=centre, ell_comps=(0.0, 0.0))
        if ra > rs:
            ra, rs = rs, ra
        self.ra = ra
        self.rs = rs
        self.b0 = b0

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.
        Faster and equivalent to Eliasdottir (2007), see Eq. A19 and Eq. A20. 

        f(R,a,s) = {R/a} / {1 + \\sqrt{1 + (R/a)^2}} - {R/s} / {1 + \\sqrt{1 + (R/s)^2}}
                 = R / {\\sqrt{a^2 + R^2} + a} - R / {\\sqrt{s^2 + R^2} + s}
                 = R * (\\sqrt{a^2 + R^2} - a) / {a^2 + R^2 - a^2} - R * (\\sqrt{s^2 + R^2} - s) / {s^2 + R^2 - s^2}
                 = (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R
        \\alpha = b0 * s / (s - a) * f(R,a,s)
        deflection_x = \\alpha * grid[:, 1] / R
                     = grid[:, 1] * b0 * s / (s - a) * (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R^2
        deflection_y = \\alpha * grid[:, 0] / R
                     = grid[:, 0] * b0 * s / (s - a) * (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R^2

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        a = self.ra
        s = self.rs
        # radii = self.radial_grid_from(grid=grid, **kwargs)
        # R2 = radii * radii
        R2 = grid[:, 1] * grid[:, 1] + grid[:, 0] * grid[:, 0]
        factor = np.sqrt(R2 + a * a) - a - np.sqrt(R2 + s * s) + s
        factor *= self.b0 * s / (s - a) / R2

        # This is in axes aligned to the major/minor axis
        deflection_x = grid[:, 1] * factor
        deflection_y = grid[:, 0] * factor

        # And here we convert back to the real axes
        return self.rotated_grid_from_reference_frame_from(
            grid=np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T), **kwargs
        )
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def analytical_hessian_2d_from(self, grid: 'aa.type.Grid2DLike', **kwargs):
        """
        Calculate the hessian matrix on a grid of (y,x) arc-second coordinates.
        Chain rule of second derivatives: 
        ∂²ψ/∂x² = ∂²ψ/∂R² * (∂R/∂x)² + ∂²R/∂x² * ∂ψ/∂R
        ∂²ψ/∂y² = ∂²ψ/∂R² * (∂R/∂y)² + ∂²R/∂y² * ∂ψ/∂R
        ∂²ψ/∂x∂y = ∂²ψ/∂R² * ∂R/∂x * ∂R/∂y + ∂²R/∂x∂y * ∂ψ/∂R


        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        grid = np.asarray(grid)
        if grid.ndim != 2 or grid.shape[1] != 2:
            raise ValueError("Grid must be a 2D array with shape (n, 2)")

        a = self.ra
        s = self.rs
        t05 = self.b0 * s / (s - a)

        # We have known the first derivatives as `deflections_yx`:
        # ∂ψ/∂R ∝ f(R,a,s) = (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R = z / R
        # ∂ψ/∂x ∝ x * (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R^2 = x * z / R^2
        # ∂ψ/∂y ∝ y * (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R^2 = y * z / R^2

        # where z = (\\sqrt{R^2 + a^2} - a - \\sqrt{R^2 + s^2} + s) / R^2

        # R = (x^2 + y^2)^(0.5)
        # ∂R/∂x = x / R
        # ∂R/∂y = y / R
        # ∂²R/∂²x = y^2 / R^3
        # ∂²R/∂²y = x^2 / R^3
        # ∂²R/∂x∂y = - x*y / R^3

        # ∂²ψ/∂²R = ∂(z/R)/∂R = (∂z/∂R * R - z * 1) / R^2
        #                     = {( R^2 / √(R^2 + a^2)) - ( R^2 / √(R^2 + s^2)) - z} / R^2
        #                     = p
        R2 = grid[:, 1] * grid[:, 1] + grid[:, 0] * grid[:, 0]
        z = np.sqrt(R2 + a * a) - a - np.sqrt(R2 + s * s) + s
        p = (1.0 - a / np.sqrt(a * a + R2)) * a / R2 - (1.0 - s / np.sqrt(s * s + R2)) * s / R2
        X = grid[:, 1] * grid[:, 1] / R2 # x^2 / R^2
        Y = grid[:, 0] * grid[:, 0] / R2 # y^2 / R^2
        XY = grid[:, 1] * grid[:, 0] / R2 # x*y / R^2

        # ∂²ψ/∂x²  = ∂²ψ/∂R² * (∂R/∂x)² + ∂²R/∂x² * ∂ψ/∂R
        #          = p * (x / R)^2 + y^2 / R^3 * z / R
        #          = p * x^2 / R^2 + z * y^2 / R^2 / R^2
        #          = p * X + z * Y / R2
        # ∂²ψ/∂y²  = ∂²ψ/∂R² * (∂R/∂y)² + ∂²R/∂y² * ∂ψ/∂R
        #          = p * (y / R)^2 + x^2 / R^3 * z / R
        #          = p * y^2 / R^2 + z * x^2 / R^2 / R^2
        #          = p * Y + z * X / R2
        # ∂²ψ/∂x∂y = ∂²ψ/∂R² * ∂R/∂x * ∂R/∂y + ∂²R/∂x∂y * ∂ψ/∂R
        #          = p * (x / R) * (y / R) + (- x*y / R^3) * z / R
        #          = p * x*y / R^2 - z * x*y / R^2 / R^2
        #          = p * XY + z * XY / R2

        # Compute Hessian matrix components
        hessian_xx = t05 * (p * X + z * Y / R2)
        hessian_xy = t05 * (p * XY - z * XY / R2)
        hessian_yx = t05 * (p * XY - z * XY / R2)
        hessian_yy = t05 * (p * Y + z * X / R2)

        return hessian_yy, hessian_xy, hessian_yx, hessian_xx

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
        `kappa_scale`. Although originally called the dPIE (Elliptical), this version
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
        ellip = np.sqrt(self.ell_comps[0] ** 2 + self.ell_comps[1] ** 2)
        MAX_ELLIP = 0.99999
        return min(ellip, MAX_ELLIP)

    def _deflection_angle(self, radii):
        """
        For a circularly symmetric dPIE profile, computes the magnitude of the deflection at each radius.
        """
        a, s = self.ra, self.rs
        radii = np.maximum(radii, 1e-8)
        f = (
            radii / (a + np.sqrt(a**2 + radii**2))
            - radii / (s + np.sqrt(s**2 + radii**2))
        )

        # c.f. Eliasdottir '07 eq. A23
        # magnitude of deflection
        # alpha = self.E0 * (s + a) / s * f
        alpha = self.b0 * s / (s - a) * f
        return alpha

    def _convergence(self, radii):
        radsq = radii * radii
        a, s = self.ra, self.rs
        # c.f. Eliasdottir '07 eqn (A3)
        # return (
        #     self.E0 / 2 * (s + a) / s *
        #     (1/np.sqrt(a**2 + radsq) - 1/np.sqrt(s**2 + radsq))
        # )
        return (
            self.b0 / 2 * s / (s - a) *
            (1/np.sqrt(a**2 + radsq) - 1/np.sqrt(s**2 + radsq))
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


class dPIEPSph(dPIEP):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        b0: float = 1.0,
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
