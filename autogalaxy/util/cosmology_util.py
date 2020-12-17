import math

import numpy as np
from astropy import constants


def arcsec_per_kpc_from(*, redshift, cosmology):
    return cosmology.arcsec_per_kpc_proper(z=redshift).value


def kpc_per_arcsec_from(*, redshift, cosmology):
    return 1.0 / cosmology.arcsec_per_kpc_proper(z=redshift).value


def angular_diameter_distance_to_earth_in_kpc_from(*, redshift, cosmology):

    angular_diameter_distance_kpc = cosmology.angular_diameter_distance(z=redshift).to(
        "kpc"
    )

    return angular_diameter_distance_kpc.value


def angular_diameter_distance_between_redshifts_in_kpc_from(
    *, redshift_0, redshift_1, cosmology
):

    angular_diameter_distance_between_redshifts_kpc = cosmology.angular_diameter_distance_z1z2(
        redshift_0, redshift_1
    ).to(
        "kpc"
    )

    return angular_diameter_distance_between_redshifts_kpc.value


def cosmic_average_density_from(*, redshift, cosmology):

    cosmic_average_density_kpc = (
        cosmology.critical_density(z=redshift).to("solMass / kpc^3").value
    )

    kpc_per_arcsec = kpc_per_arcsec_from(redshift=redshift, cosmology=cosmology)

    return cosmic_average_density_kpc * kpc_per_arcsec ** 3.0


def cosmic_average_density_solar_mass_per_kpc3_from(*, redshift, cosmology):

    cosmic_average_density_kpc = (
        cosmology.critical_density(z=redshift).to("solMass / kpc^3").value
    )

    return cosmic_average_density_kpc


def critical_surface_density_between_redshifts_from(
    *, redshift_0, redshift_1, cosmology
):

    critical_surface_density_kpc = critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        redshift_0=redshift_0, redshift_1=redshift_1, cosmology=cosmology
    )

    kpc_per_arcsec = kpc_per_arcsec_from(redshift=redshift_0, cosmology=cosmology)

    return critical_surface_density_kpc * kpc_per_arcsec ** 2.0


def critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
    *, redshift_0, redshift_1, cosmology
):

    const = constants.c.to("kpc / s") ** 2.0 / (
        4 * math.pi * constants.G.to("kpc3 / (solMass s2)")
    )

    angular_diameter_distance_of_redshift_0_to_earth_kpc = angular_diameter_distance_to_earth_in_kpc_from(
        redshift=redshift_0, cosmology=cosmology
    )

    angular_diameter_distance_of_redshift_1_to_earth_kpc = angular_diameter_distance_to_earth_in_kpc_from(
        redshift=redshift_1, cosmology=cosmology
    )

    angular_diameter_distance_between_redshifts_kpc = angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=redshift_0, redshift_1=redshift_1, cosmology=cosmology
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
    *, redshift_0, redshift_1, redshift_final, cosmology
):

    angular_diameter_distance_between_redshifts_0_and_1 = (
        cosmology.angular_diameter_distance_z1z2(z1=redshift_0, z2=redshift_1)
        .to("kpc")
        .value
    )

    angular_diameter_distance_to_redshift_final = (
        cosmology.angular_diameter_distance(z=redshift_final).to("kpc").value
    )

    angular_diameter_distance_of_redshift_1_to_earth = (
        cosmology.angular_diameter_distance(z=redshift_1).to("kpc").value
    )

    angular_diameter_distance_between_redshift_1_and_final = (
        cosmology.angular_diameter_distance_z1z2(z1=redshift_0, z2=redshift_final)
        .to("kpc")
        .value
    )

    return (
        angular_diameter_distance_between_redshifts_0_and_1
        * angular_diameter_distance_to_redshift_final
    ) / (
        angular_diameter_distance_of_redshift_1_to_earth
        * angular_diameter_distance_between_redshift_1_and_final
    )


def velocity_dispersion_from(*, redshift_0, redshift_1, einstein_radius, cosmology):

    const = constants.c.to("kpc / s")

    angular_diameter_distance_to_redshift_0_kpc = angular_diameter_distance_to_earth_in_kpc_from(
        redshift=redshift_1, cosmology=cosmology
    )

    angular_diameter_distance_to_redshift_1_kpc = angular_diameter_distance_to_earth_in_kpc_from(
        redshift=redshift_1, cosmology=cosmology
    )

    angular_diameter_distance_between_redshifts_kpc = angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=redshift_0, redshift_1=redshift_1, cosmology=cosmology
    )

    kpc_per_arcsec = kpc_per_arcsec_from(redshift=redshift_0, cosmology=cosmology)

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
