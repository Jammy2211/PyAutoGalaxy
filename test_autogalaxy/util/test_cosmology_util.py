import autogalaxy as ag
import pytest
from astropy import cosmology as cosmo
import numpy as np

planck = cosmo.Planck15


def test__arcsec_to_kpc_conversion():

    arcsec_per_kpc = ag.util.cosmology.arcsec_per_kpc_from(
        redshift=0.1, cosmology=planck
    )

    assert arcsec_per_kpc == pytest.approx(0.525060, 1e-5)

    kpc_per_arcsec = ag.util.cosmology.kpc_per_arcsec_from(
        redshift=0.1, cosmology=planck
    )

    assert kpc_per_arcsec == pytest.approx(1.904544, 1e-5)

    arcsec_per_kpc = ag.util.cosmology.arcsec_per_kpc_from(
        redshift=1.0, cosmology=planck
    )

    assert arcsec_per_kpc == pytest.approx(0.1214785, 1e-5)

    kpc_per_arcsec = ag.util.cosmology.kpc_per_arcsec_from(
        redshift=1.0, cosmology=planck
    )

    assert kpc_per_arcsec == pytest.approx(8.231907, 1e-5)


def test__angular_diameter_distances():

    angular_diameter_distance_to_earth_kpc = ag.util.cosmology.angular_diameter_distance_to_earth_in_kpc_from(
        redshift=0.1, cosmology=planck
    )

    assert angular_diameter_distance_to_earth_kpc == pytest.approx(392840, 1e-5)

    angular_diameter_distance_between_redshifts_kpc = ag.util.cosmology.angular_diameter_distance_between_redshifts_in_kpc_from(
        redshift_0=0.1, redshift_1=1.0, cosmology=planck
    )

    assert angular_diameter_distance_between_redshifts_kpc == pytest.approx(
        1481890.4, 1e-5
    )


def test__cosmic_average_densities_solar_mass_per_kpc3():

    cosmic_average_density = ag.util.cosmology.cosmic_average_density_from(
        redshift=0.6, cosmology=planck
    )

    assert cosmic_average_density == pytest.approx(81280.09116133313, 1.0e-4)

    cosmic_average_density = ag.util.cosmology.cosmic_average_density_solar_mass_per_kpc3_from(
        redshift=0.6, cosmology=planck
    )

    assert cosmic_average_density == pytest.approx(249.20874, 1.0e-4)


def test__critical_surface_mass_densities():

    critical_surface_density = ag.util.cosmology.critical_surface_density_between_redshifts_from(
        redshift_0=0.1, redshift_1=1.0, cosmology=planck
    )

    assert critical_surface_density == pytest.approx(17593241668, 1e-2)

    critical_surface_density = ag.util.cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        redshift_0=0.1, redshift_1=1.0, cosmology=planck
    )

    assert critical_surface_density == pytest.approx(4.85e9, 1e-2)


def test__velocity_dispersion_from():

    velocity_dispersion = ag.util.cosmology.velocity_dispersion_from(
        redshift_0=0.5, redshift_1=1.0, einstein_radius=1.0, cosmology=planck
    )

    assert velocity_dispersion == pytest.approx(249.03449, 1.0e-4)

    velocity_dispersion = ag.util.cosmology.velocity_dispersion_from(
        redshift_0=0.5, redshift_1=1.0, einstein_radius=2.0, cosmology=planck
    )

    assert velocity_dispersion == pytest.approx(np.sqrt(2) * 249.03449, 1.0e-4)
