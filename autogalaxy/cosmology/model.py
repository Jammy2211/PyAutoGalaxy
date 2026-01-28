import numpy as np
import math


class LensingCosmology:
    """
    Class containing specific functions for performing gravitational lensing cosmology calculations.

    This version is JAX-compatible by using an explicit `xp` backend (NumPy or jax.numpy).
    """

    def arcsec_per_kpc_proper(self, z: float, xp=np) -> float:
        """
        Angular separation in arcsec corresponding to 1 proper kpc at redshift z.

        This matches astropy.cosmology.arcsec_per_kpc_proper.

        Proper transverse distance uses the angular diameter distance D_A(z):

            arcsec_per_kpc_proper = 206265 / D_A(z)

        where D_A(z) is in kpc.
        """

        angular_diameter_distance_kpc = self.angular_diameter_distance_kpc_z1z2(
            0.0, z, xp=xp
        )

        return xp.asarray(206265.0) / angular_diameter_distance_kpc

    def kpc_proper_per_arcsec(self, z: float, xp=np) -> float:
        """
        Proper transverse separation in kpc corresponding to 1 arcsec at redshift z.

        This matches the inverse of astropy.cosmology.arcsec_per_kpc_proper:

            kpc_proper_per_arcsec = D_A(z) / 206265
        """

        angular_diameter_distance_kpc = self.angular_diameter_distance_kpc_z1z2(
            0.0, z, xp=xp
        )

        return angular_diameter_distance_kpc / xp.asarray(206265.0)

    def arcsec_per_kpc_from(self, redshift: float, xp=np) -> float:
        """
        Angular separation in arcsec corresponding to a proper kpc at redshift `z`.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.

        This is a thin convenience wrapper around `arcsec_per_kpc_proper`.
        """
        return self.arcsec_per_kpc_proper(z=redshift, xp=xp)

    def kpc_per_arcsec_from(self, redshift: float, xp=np) -> float:
        """
        Separation in transverse proper kpc corresponding to an arcsec at redshift `z`.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.

        This is a thin convenience wrapper around `kpc_proper_per_arcsec`.
        """
        return self.kpc_proper_per_arcsec(z=redshift, xp=xp)

    def angular_diameter_distance_to_earth_in_kpc_from(self, redshift: float, xp=np) -> float:
        """
        Angular diameter distance from the input `redshift` to redshift zero (e.g. us, the observer on earth) in
        kiloparsecs.

        This gives the proper (sometimes called 'physical') transverse distance corresponding to an angle of 1 radian
        for an object at redshift `z`.

        Weinberg, 1972, pp 421-424; Weedman, 1986, pp 65-67; Peebles, 1993, pp 325-327.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.
        This function therefore returns only the value of the astropy function it wraps, omitting the units instance.

        Parameters
        ----------
        redshift
            Input redshift from which the angular diameter distance to Earth is calculated.
        """
        return self.angular_diameter_distance_kpc_z1z2(0.0, redshift, xp=xp)

    def angular_diameter_distance_between_redshifts_in_kpc_from(
        self, redshift_0: float, redshift_1: float, xp=np
    ) -> float:
        """
        Angular diameter distance from an input `redshift_0` to another input `redshift_1`.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.
        This function therefore returns only the value of the astropy function it wraps, omitting the units instance.

        Parameters
        ----------
        redshift_0
            Redshift from which the angular diameter distance to the other redshift is calculated.
        redshift_1
            Redshift from which the angular diameter distance to the other redshift is calculated.
        """
        return self.angular_diameter_distance_kpc_z1z2(
            redshift_0, redshift_1, xp=xp
        )

    def cosmic_average_density_from(self, redshift: float) -> float:
        """
        Critical density of the Universe at an input `redshift` in units of solar masses.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.
        This function therefore returns only the value of the astropy function it wraps, omitting the units instance.

        Parameters
        ----------
        redshift
            Redshift at which the critiical density in solMass of the Universe is calculated.
        """

        cosmic_average_density_kpc = (
            self.critical_density(z=redshift).to("solMass / kpc^3").value
        )

        kpc_per_arcsec = self.kpc_per_arcsec_from(redshift=redshift)

        return cosmic_average_density_kpc * kpc_per_arcsec**3.0

    def cosmic_average_density_solar_mass_per_kpc3_from(self, redshift: float) -> float:
        """
        Critical density of the Universe at an input `redshift` in units of solar masses per kiloparsecs**3.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.
        This function therefore returns only the value of the astropy function it wraps, omitting the units instance.

        Parameters
        ----------
        redshift
            Redshift at which the critiical density in solMass/kpc^3 of the Universe is calculated.
        """
        cosmic_average_density_kpc = (
            self.critical_density(z=redshift).to("solMass / kpc^3").value
        )

        return cosmic_average_density_kpc

    def critical_surface_density_between_redshifts_from(
        self, redshift_0: float, redshift_1: float
    ) -> float:
        """
        The critical surface density for lensing, often written as $\sigma_{cr}$, is given by:

        critical_surface_density = (c^2 * D_s) / (4 * pi * G * D_ls * D_l)

        c = speed of light
        G = Newton's gravity constant
        D_s = angular_diameter_distance_of_source_redshift_to_earth
        D_ls = angular_diameter_distance_of_lens_redshift_to_source_redshift
        D_l = angular_diameter_distance_of_lens_redshift_to_earth

        This function returns the critical surface density in units of solar masses, which are convenient units for
        converting the inferred masses of a model from angular units (e.g. dimensionless units inferred from
        data in arcseconds) to solar masses.

        Parameters
        ----------
        redshift_0
            The redshift of the first strong lens galaxy (E.g. the lens galaxy) for which the critical surface
            density is calculated.
        redshift_1
            The redshift of the second strong lens galaxy (E.g. the lens galaxy) for which the critical surface
            density is calculated.
        """
        critical_surface_density_kpc = (
            self.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
                redshift_0=redshift_0, redshift_1=redshift_1
            )
        )

        kpc_per_arcsec = self.kpc_per_arcsec_from(redshift=redshift_0)

        return critical_surface_density_kpc * kpc_per_arcsec**2.0

    def critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        self, redshift_0: float, redshift_1: float
    ) -> float:
        """
        The critical surface density for lensing, often written as $\sigma_{cr}$, is given by:

        critical_surface_density = (c^2 * D_s) / (4 * pi * G * D_ls * D_l)

        c = speed of light
        G = Newton's gravity constant
        D_s = Angular diameter distance of source redshift to earth
        D_ls = Angular diameter distance of lens redshift to source redshift
        D_l = Angular diameter distance of lens redshift to earth

        This function returns the critical surface density in units of solar masses / kpc^2, which are convenient
        units for converting the inferred masses of a model from angular units (e.g. dimensionless units inferred
        from data in arcseconds) to solar masses.

        Parameters
        ----------
        redshift_0
            The redshift of the first strong lens galaxy (E.g. the lens galaxy) for which the critical surface
            density is calculated.
        redshift_1
            The redshift of the second strong lens galaxy (E.g. the lens galaxy) for which the critical surface
            density is calculated.
        """
        from astropy import constants

        const = constants.c.to("kpc / s") ** 2.0 / (
            4 * math.pi * constants.G.to("kpc3 / (solMass s2)")
        )

        angular_diameter_distance_of_redshift_0_to_earth_kpc = (
            self.angular_diameter_distance_to_earth_in_kpc_from(redshift=redshift_0)
        )

        angular_diameter_distance_of_redshift_1_to_earth_kpc = (
            self.angular_diameter_distance_to_earth_in_kpc_from(redshift=redshift_1)
        )

        angular_diameter_distance_between_redshifts_kpc = (
            self.angular_diameter_distance_between_redshifts_in_kpc_from(
                redshift_0=redshift_0, redshift_1=redshift_1
            )
        )

        return (
            const
            * angular_diameter_distance_of_redshift_1_to_earth_kpc
            / (
                angular_diameter_distance_between_redshifts_kpc
                * angular_diameter_distance_of_redshift_0_to_earth_kpc
            )
        ).value

    def scaling_factor_between_redshifts_from(
        self, redshift_0: float, redshift_1: float, redshift_final: float
    ) -> float:
        """
        For strong lens systems with more than 2 planes, the deflection angles between different planes must be scaled
        by the angular diameter distances between the planes in order to properly perform multi-plane ray-tracing. This
        function computes the factor to scale deflections between `redshift_0` and `reshift_final`, to deflections between
        `redshift_0` and `redshift_1`.

        The second redshift should be strictly larger than the first. The scaling factor is unity when `redshift_1`
        is `redshift_final`, and 0 when `redshift_0` is equal to `redshift_1`.

        For a system with a first lens galaxy l0 at `redshift_0`, second lens galaxy l1 at `redshift_1` and final
        source galaxy at `redshift_final` this scaling factor is given by:

        (D_l0l1 * D_s) / (D_l1* D_l0s)

        The critical surface density for lensing, often written as $\\sigma_{cr}$, is given by:

        critical_surface_density = (c^2 * D_s) / (4 * pi * G * D_ls * D_l)

        D_l0l1 = Angular diameter distance of first lens redshift to second lens redshift.
        D_s = Angular diameter distance of source redshift to earth
        D_l1 = Angular diameter distance of second lens redshift to Earth.
        D_l0s = Angular diameter distance of first lens redshift to source redshift

        For systems with more planes this scaling factor is computed multiple times for the different redshift
        combinations and applied recursively when scaling the deflection angles.

        Parameters
        ----------
        redshift_0
            The redshift of the first strong lens galaxy.
        redshift_1
            The redshift of the second strong lens galaxy.
        redshift_final
            The redshift of the source galaxy.
        """
        angular_diameter_distance_between_redshifts_0_and_1 = (
            self.angular_diameter_distance_z1z2(z1=redshift_0, z2=redshift_1)
            .to("kpc")
            .value
        )

        angular_diameter_distance_to_redshift_final = (
            self.angular_diameter_distance(z=redshift_final).to("kpc").value
        )

        angular_diameter_distance_of_redshift_1_to_earth = (
            self.angular_diameter_distance(z=redshift_1).to("kpc").value
        )

        angular_diameter_distance_between_redshift_0_and_final = (
            self.angular_diameter_distance_z1z2(z1=redshift_0, z2=redshift_final)
            .to("kpc")
            .value
        )

        return (
            angular_diameter_distance_between_redshifts_0_and_1
            * angular_diameter_distance_to_redshift_final
        ) / (
            angular_diameter_distance_of_redshift_1_to_earth
            * angular_diameter_distance_between_redshift_0_and_final
        )

    def velocity_dispersion_from(
        self, redshift_0: float, redshift_1: float, einstein_radius: float
    ) -> float:
        """
        For a strong lens galaxy with an Einstien radius in arcseconds, the corresponding velocity dispersion of the
        lens galaxy can be computed (assuming an isothermal mass distribution).

        The velocity dispersion is given by:

        velocity dispersion = (c * R_Ein * D_s) / (4 * pi * D_l * D_ls)

        c = speed of light
        D_s = Angular diameter distance of source redshift to earth
        D_ls = Angular diameter distance of lens redshift to source redshift
        D_l = Angular diameter distance of lens redshift to earth

        Parameters
        ----------
        redshift_0
            The redshift of the first strong lens galaxy (the lens).
        redshift_1
            The redshift of the second strong lens galaxy (the source).
        """
        from astropy import constants

        const = constants.c.to("kpc / s")

        angular_diameter_distance_to_redshift_0_kpc = (
            self.angular_diameter_distance_to_earth_in_kpc_from(redshift=redshift_1)
        )

        angular_diameter_distance_to_redshift_1_kpc = (
            self.angular_diameter_distance_to_earth_in_kpc_from(redshift=redshift_1)
        )

        angular_diameter_distance_between_redshifts_kpc = (
            self.angular_diameter_distance_between_redshifts_in_kpc_from(
                redshift_0=redshift_0, redshift_1=redshift_1
            )
        )

        kpc_per_arcsec = self.kpc_per_arcsec_from(redshift=redshift_0)

        einstein_radius_kpc = einstein_radius * kpc_per_arcsec

        velocity_dispersion_kpc = const * np.sqrt(
            (einstein_radius_kpc * angular_diameter_distance_to_redshift_1_kpc)
            / (
                4
                * np.pi
                * angular_diameter_distance_to_redshift_0_kpc
                * angular_diameter_distance_between_redshifts_kpc
            )
        )

        return velocity_dispersion_kpc.to("km/s").value


class FlatLambdaCDM(LensingCosmology):
    def __init__(
        self,
        H0: float = 67.66,
        Om0: float = 0.30966,
        Tcmb0: float = 2.7255,
        Neff: float = 3.046,
        m_nu: float = 0.0,
        Ob0: float = 0.04897,
    ):
        """
        A wrapper for the astropy `FlatLambdaCDM` cosmology class, which allows it to be used for modeling such
        that the cosmological parameters are free parameters which can be fitted for.

        The interface of this class is the same as the astropy `FlatLambdaCDM` class, it simply overwrites the
        __init__ method and inherits from it in a way that ensures **PyAutoFit** can compose a model from it
        without issue.

        The class also inherits from `LensingCosmology`, which is a class that provides additional functionality
        for calculating lensing specific quantities in the cosmology.

        Parameters
        ----------
        H0
            The Hubble constant at z=0.
        Om0
            The total matter density at z=0.
        Tcmb0
            The CMB temperature at z=0.
        Neff
            The effective number of neutrinos.
        m_nu
            The sum of the neutrino masses.
        Ob0
            The baryon density at z=0.
        """
        self.H0 = H0
        self.Om0 = Om0
        self.Tcmb0 = Tcmb0
        self.Neff = Neff
        self.m_nu = m_nu
        self.Ob0 = Ob0

        # Make ΛCDM a special case of wCDM
        self.w0 = -1.0

    @staticmethod
    def _simpson_1d(y, x, xp=np):
        """
        Composite Simpson's rule on a 1D grid.

        Requirements:
        - x is 1D, evenly spaced
        - len(x) is odd  (i.e. number of intervals is even)

        Works with numpy or jax.numpy passed as xp.
        """
        n = x.shape[0]
        if (n % 2) == 0:
            raise ValueError("Simpson's rule requires an odd number of samples (even number of intervals).")

        h = (x[-1] - x[0]) / (n - 1)

        # y0 + yN
        s = y[0] + y[-1]

        # 4 * sum of odd indices
        s = s + 4.0 * xp.sum(y[1:-1:2])

        # 2 * sum of even indices (excluding endpoints)
        s = s + 2.0 * xp.sum(y[2:-1:2])

        return (h / 3.0) * s

    def angular_diameter_distance_kpc_z1z2(
        self,
        z1: float,
        z2: float,
        n_steps: int = 8193,  # odd by default
        xp=np,
    ):
        """
        D_A(z1,z2) in kpc for flat wCDM using Simpson's rule.

        Includes:
        - photons via Tcmb0
        - *massive neutrinos* via m_nu (matter-like, Omega_nu h^2 = sum(m_nu)/93.14 eV)

        Notes:
        - Flat universe: Omega_k = 0
        - Dark energy equation of state constant w0
        - This is designed to better match astropy's Planck15-style backgrounds.
        """
        # Ensure odd number of samples for Simpson (safe: n_steps is a Python int)
        if (n_steps % 2) == 0:
            n_steps += 1

        z1a = xp.asarray(z1)
        z2a = xp.asarray(z2)
        same = z1a == z2a

        c_km_s = xp.asarray(299792.458)

        H0 = xp.asarray(self.H0)
        h = H0 / xp.asarray(100.0)

        Om0 = xp.asarray(self.Om0)
        w0 = xp.asarray(self.w0)

        # ---- Photon radiation density today ----
        # Omega_gamma * h^2 ≈ 2.469e-5 * (Tcmb/2.7255)^4
        Tcmb = xp.asarray(self.Tcmb0)
        Ogamma_h2 = xp.asarray(2.469e-5) * (Tcmb / xp.asarray(2.7255)) ** 4
        Ogamma0 = Ogamma_h2 / (h * h)

        # ---- Massive neutrinos (matter-like approximation) ----
        # Omega_nu h^2 ≈ sum(m_nu)/93.14 eV
        m_nu = getattr(self, "m_nu", 0.0)
        m_nu_sum = xp.sum(xp.asarray(m_nu))  # works if m_nu is float or array-like
        Onu_h2 = m_nu_sum / xp.asarray(93.14)
        Onu0 = Onu_h2 / (h * h)

        Neff = xp.asarray(self.Neff)
        Onu_rad0 = Ogamma0 * xp.asarray(0.2271) * Neff
        Or0 = Ogamma0 + Onu_rad0

        # Flatness: Omega_de = 1 - Omega_m - Omega_r - Omega_nu
        Ode0 = xp.asarray(1.0) - Om0 - Or0 - Onu0

        def E(z):
            zp1 = xp.asarray(1.0) + z
            # E^2 = (Om+Onu_m)(1+z)^3 + Or(1+z)^4 + Ode(1+z)^{3(1+w)}
            return xp.sqrt(
                (Om0 + Onu0) * zp1 ** 3
                + Or0 * zp1 ** 4
                + Ode0 * zp1 ** (xp.asarray(3.0) * (xp.asarray(1.0) + w0))
            )

        z_grid = xp.linspace(z1a, z2a, n_steps)
        integrand = 1.0 / E(z_grid)

        integral = self._simpson_1d(integrand, z_grid, xp=xp)

        Dc_Mpc = (c_km_s / H0) * integral
        Da_Mpc = Dc_Mpc / (xp.asarray(1.0) + z2a)
        Da_kpc = Da_Mpc * xp.asarray(1.0e3)

        return xp.where(same, xp.asarray(0.0), Da_kpc)


class Planck15(FlatLambdaCDM):

    def __init__(self):

        super().__init__(
            H0=67.74,
            Om0=0.3075,
            Tcmb0=2.7255,
            Neff=3.046,
            m_nu=0.06,
            Ob0=0.0486,
        )
