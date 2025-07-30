from typing import Tuple
import numpy as np

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

class PIEMD(MassProfile): 

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        b0: float = 1.0,
    ):
        super().__init__(centre=centre, ell_comps=ell_comps)

        self.ra = ra
        self.b0 = b0

    def _ci05(self,x, y, eps, rc):
        print('x:',x,'y',y)
        # Calculate intermediate values
        if eps < 1e-10:
            eps = 1e-10
        sqe = np.sqrt(eps)
        cx1 = (1.0 - eps) / (1.0 + eps)
        cxro = (1.0 + eps) * (1.0 + eps)
        cyro = (1.0 - eps) * (1.0 - eps)
        rem2 = x * x / cxro + y * y / cyro
        
        # Define complex numbers
        zci = np.complex128(complex(0.0, -0.5 * (1.0 - eps * eps) / sqe))
        znum = np.complex128(cx1 * x + 1j * (2.0 * sqe * np.sqrt(rc * rc + rem2) - y / cx1))
        zden = np.complex128(x + 1j * (2.0 * rc * sqe - y))
            
        # Perform complex division: zis = znum / zden
        norm = zden.real * zden.real + zden.imag * zden.imag  # |zden|^2
        if np.any(norm < 1e-10):
            raise ValueError("zden magnitude too small, risk of division by zero")
        zis_re = (znum.real * zden.real + znum.imag * zden.imag) / norm
        zis_im = (znum.imag * zden.real - znum.real * zden.imag) / norm
        zis = np.complex128(zis_re + 1j * zis_im)
        
        # Calculate ln(zis) = ln(|zis|) + i*Arg(zis)
        zis_mag = np.abs(zis)
        if np.any(zis_mag < 1e-10):
            raise ValueError("zis magnitude too small, risk of log(0)")
        zis_re = np.log(zis_mag)
        zis_im = np.angle(zis)
        zis = np.complex128(zis_re + 1j * zis_im)
        
        # Calculate zres = zci * ln(zis)
        zres = zci * zis
        
        return zres
    
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

        zis = self._ci05(
            x=grid[:, 1], y=grid[:, 0], eps=ellip, rc=self.ra
        )

        # This is in axes aligned to the major/minor axis
        deflection_x = self.b0 * zis.real
        deflection_y = self.b0 * zis.imag

        # And here we convert back to the real axes
        return self.rotated_grid_from_reference_frame_from(
            grid=np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T), **kwargs
        )

class dPIEMD(MassProfile):

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.0,
        rs: float = 2.0,
        b0: float = 1.0,
    ):
        super().__init__(centre=centre, ell_comps=ell_comps)

        if ra > rs:
            ra, rs = rs, ra

        self.ra = ra
        self.rs = rs
        self.b0 = b0

    def _ci05f(self, x, y, eps, rc, rcut):
        # Prevent division by zero by setting a minimum value for eps
        if eps < 1e-10:
            eps = 1e-10
        
        # Calculate intermediate values
        sqe = np.sqrt(eps)  # Square root of ellipticity
        cx1 = (1.0 - eps) / (1.0 + eps)  # Scaling factor for x
        cxro = (1.0 + eps) * (1.0 + eps)  # Denominator for x^2 term
        cyro = (1.0 - eps) * (1.0 - eps)  # Denominator for y^2 term
        rem2 = x * x / cxro + y * y / cyro  # Elliptical radius term

        # Define zci (complex constant)
        zci = np.complex128(complex(0.0, -0.5 * (1.0 - eps * eps) / sqe))

        # Compute complex numbers for rc
        znum_rc = np.complex128(cx1 * x + 1j * (2.0 * sqe * np.sqrt(rc * rc + rem2) - y / cx1))
        zden_rc = np.complex128(x + 1j * (2.0 * rc * sqe - y))

        # Compute complex numbers for rcut
        znum_rcut = np.complex128(cx1 * x + 1j * (2.0 * sqe * np.sqrt(rcut * rcut + rem2) - y / cx1))
        zden_rcut = np.complex128(x + 1j * (2.0 * rcut * sqe - y))

        # Compute zis_rc / zis_rcut = (znum_rc / zden_rc) / (znum_rcut / zden_rcut)
        # Using manual complex division: (aa + bb*i) / (cc + dd*i)
        aa = znum_rc.real * zden_rc.real - znum_rc.imag * zden_rcut.imag
        bb = znum_rc.real * zden_rcut.imag + znum_rc.imag * zden_rc.real
        cc = znum_rc.real * zden_rc.real - zden_rc.imag * znum_rcut.imag
        dd = znum_rc.real * zden_rc.imag + zden_rc.real * znum_rcut.imag

        norm = cc * cc + dd * dd
        if np.any(norm < 1e-10):
            raise ValueError("Denominator magnitude too small in zis_rc/zis_rcut, risk of division by zero")
        # Compute real and imaginary parts of zis_rc / zis_rcut
        aaa = (aa * cc + bb * dd) / norm
        bbb = (bb * cc - aa * dd) / norm

        # Compute zr = log(zis_rc / zis_rcut) = log(aaa + bbb*i)
        norm2 = aaa * aaa + bbb * bbb
        if np.any(norm2 < 1e-10):
            raise ValueError("zr magnitude too small, risk of log(0)")
        zr_re = np.log(np.sqrt(norm2))  # Real part: log(|zis_rc / zis_rcut|)
        zr_im = np.arctan2(bbb, aaa)  # Imaginary part: phase of zis_rc / zis_rcut
        zr = np.complex128(zr_re + 1j * zr_im)

        # Compute final result: zres = zci * zr
        zres = zci * zr

        return zres
    
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
        t05 = self.b0 * self.rs / (self.rs - self.ra)
        zis = self._ci05f(
            x=grid[:, 1], y=grid[:, 0], eps=ellip, rc=self.ra, rcut=self.rs
        )

        # This is in axes aligned to the major/minor axis
        deflection_x = t05 * zis.real
        deflection_y = t05 * zis.imag

        # And here we convert back to the real axes
        return self.rotated_grid_from_reference_frame_from(
            grid=np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T), **kwargs
        )
    
class dPIEMDSph(dPIEMD):

    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        b0: float = 1.0
    ):
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
        X = self.ra
        Y = self.rs
        u = grid[:, 1] * grid[:, 1] + grid[:, 0] * grid[:, 0]
        t05 = np.sqrt(u + X * X) - X - np.sqrt(u + Y * Y) +Y
        t05 *= self.b0 * Y / (Y - X) / u

        # This is in axes aligned to the major/minor axis
        deflection_x = t05 * grid[:, 1]
        deflection_y = t05 * grid[:, 0]

        # And here we convert back to the real axes
        return self.rotated_grid_from_reference_frame_from(
            grid=np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T), **kwargs
        )