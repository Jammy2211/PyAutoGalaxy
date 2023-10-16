from astropy import constants
from astropy import cosmology as cosmo
import math
import numpy as np


class LensingCosmology(cosmo.FLRW):
    """
    Class containing specific functions for performing gravitational lensing cosmology calculations.

    By inheriting from the astropy `cosmo.FLRW` class this provides many additional methods for performing cosmological
    calculations.
    """

    def arcsec_per_kpc_from(self, redshift: float) -> float:
        """
        Angular separation in arcsec corresponding to a proper kpc at redshift `z`.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.
        This function therefore returns only the value of the astropy function it wraps, omitting the units instance.

        Parameters
        ----------
        redshift
            Input redshift from which the angular separation is calculated at.
        """
        return self.arcsec_per_kpc_proper(z=redshift).value

    def kpc_per_arcsec_from(self, redshift: float) -> float:
        """
        Separation in transverse proper kpc corresponding to an arcminute at redshift `z`.

        For simplicity, **PyAutoLens** internally uses only certain units to perform lensing cosmology calculations.
        This function therefore returns only the value of the astropy function it wraps, omitting the units instance.

        Parameters
        ----------
        redshift
            Input redshift from which the transverse proper kpc value is calculated at.
        """
        return 1.0 / self.arcsec_per_kpc_proper(z=redshift).value

    def angular_diameter_distance_to_earth_in_kpc_from(self, redshift: float) -> float:
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
        angular_diameter_distance_kpc = self.angular_diameter_distance(z=redshift).to(
            "kpc"
        )

        return angular_diameter_distance_kpc.value

    def angular_diameter_distance_between_redshifts_in_kpc_from(
        self, redshift_0: float, redshift_1: float
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
        angular_diameter_distance_between_redshifts_kpc = (
            self.angular_diameter_distance_z1z2(redshift_0, redshift_1).to("kpc")
        )

        return angular_diameter_distance_between_redshifts_kpc.value

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
