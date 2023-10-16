from astropy import units

import numpy as np
import warnings

from autogalaxy.cosmology.wrap import Planck15


def kappa_s_and_scale_radius_for_duffy(mass_at_200, redshift_object, redshift_source):
    """
    Computes the AutoGalaxy NFW parameters (kappa_s, scale_radius) for an NFW halo of the given
    mass, enforcing the Duffy '08 mass-concentration relation.

    Interprets mass as *`M_{200c}`*, not `M_{200m}`.
    """
    cosmology = Planck15()

    cosmic_average_density = (
        cosmology.critical_density(redshift_object).to(units.solMass / units.kpc**3)
    ).value

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object, redshift_1=redshift_source
        )
    )

    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object)

    radius_at_200 = (
        mass_at_200 / (200.0 * cosmic_average_density * (4.0 * np.pi / 3.0))
    ) ** (
        1.0 / 3.0
    )  # r200
    coefficient = 5.71 * (1.0 + redshift_object) ** (
        -0.47
    )  # The coefficient of Duffy mass-concentration (Duffy+2008)
    concentration = coefficient * (mass_at_200 / 2.952465309e12) ** (
        -0.084
    )  # mass-concentration relation. (Duffy+2008)
    de_c = (
        200.0
        / 3.0
        * (
            concentration**3
            / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
        )
    )  # rho_c

    scale_radius_kpc = radius_at_200 / concentration  # scale radius in kpc
    rho_s = cosmic_average_density * de_c  # rho_s
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density  # kappa_s
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # scale radius in arcsec

    return kappa_s, scale_radius, radius_at_200


def kappa_s_and_scale_radius_for_ludlow(
    mass_at_200, scatter_sigma, redshift_object, redshift_source
):
    """
    Computes the AutoGalaxy NFW parameters (kappa_s, scale_radius) for an NFW halo of the given
    mass, enforcing the Ludlow '16 mass-concentration relation.

    Interprets mass as *`M_{200c}`*, not `M_{200m}`.
    """
    from colossus.cosmology import cosmology as col_cosmology
    from colossus.halo.concentration import concentration as col_concentration

    warnings.filterwarnings("ignore")

    cosmology = Planck15()

    col_cosmo = col_cosmology.setCosmology("planck15")
    m_input = mass_at_200 * col_cosmo.h
    concentration = col_concentration(
        m_input, "200c", redshift_object, model="ludlow16"
    )

    concentration = 10.0 ** (np.log10(concentration) + scatter_sigma * 0.15)

    cosmic_average_density = (
        cosmology.critical_density(redshift_object).to(units.solMass / units.kpc**3)
    ).value

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object, redshift_1=redshift_source
        )
    )

    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object)

    radius_at_200 = (
        mass_at_200 / (200.0 * cosmic_average_density * (4.0 * np.pi / 3.0))
    ) ** (
        1.0 / 3.0
    )  # r200

    de_c = (
        200.0
        / 3.0
        * (
            concentration**3
            / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
        )
    )  # rho_c

    scale_radius_kpc = radius_at_200 / concentration  # scale radius in kpc
    rho_s = cosmic_average_density * de_c  # rho_s
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density  # kappa_s
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # scale radius in arcsec

    return kappa_s, scale_radius, radius_at_200
