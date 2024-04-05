import numpy as np
import pytest

import autogalaxy as ag


def test__coord_function_f__from():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
    )

    # r > 1

    coord_f = mp.coord_func_f(grid_radius=np.array([2.0, 3.0]))

    assert coord_f == pytest.approx(np.array([0.604599, 0.435209]), 1.0e-4)

    # r < 1

    coord_f = mp.coord_func_f(grid_radius=np.array([0.5, 1.0 / 3.0]))

    assert coord_f == pytest.approx(1.52069, 1.86967, 1.0e-4)
    #
    # r == 1

    coord_f = mp.coord_func_f(grid_radius=np.array([1.0, 1.0]))

    assert (coord_f == np.array([1.0, 1.0])).all()


def test__coord_function_g__from():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
    )

    # r > 1

    coord_g = mp.coord_func_g(grid_radius=np.array([2.0, 3.0]))

    assert coord_g == pytest.approx(np.array([0.13180, 0.070598]), 1.0e-4)

    # r < 1

    coord_g = mp.coord_func_g(grid_radius=np.array([0.5, 1.0 / 3.0]))

    assert coord_g == pytest.approx(np.array([0.69425, 0.97838]), 1.0e-4)

    # r == 1

    coord_g = mp.coord_func_g(grid_radius=np.array([1.0, 1.0]))

    assert coord_g == pytest.approx(np.real(np.array([1.0 / 3.0, 1.0 / 3.0])), 1.0e-4)


def test__coord_function_h__from():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
    )

    coord_h = mp.coord_func_h(grid_radius=np.array([0.5, 3.0]))

    assert coord_h == pytest.approx(np.array([0.134395, 0.840674]), 1.0e-4)


def test__coord_function_k__from():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=2.0
    )

    coord_k = mp.coord_func_k(grid_radius=np.array([2.0, 3.0]))

    assert coord_k == pytest.approx(np.array([-0.09983408, -0.06661738]), 1.0e-4)

    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=4.0
    )

    coord_k = mp.coord_func_k(grid_radius=np.array([2.0, 3.0]))

    assert coord_k == pytest.approx(np.array([-0.19869011, -0.1329414]), 1.0e-4)


def test__coord_function_l__from():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=2.0
    )

    coord_l = mp.coord_func_l(grid_radius=np.array([2.0, 2.0]))

    assert coord_l == pytest.approx(np.array([0.00080191, 0.00080191]), 1.0e-4)

    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
    )

    coord_l = mp.coord_func_l(grid_radius=np.array([2.0, 2.0]))

    assert coord_l == pytest.approx(np.array([0.00178711, 0.00178711]), 1.0e-4)

    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
    )

    coord_l = mp.coord_func_l(grid_radius=np.array([3.0, 3.0]))

    assert coord_l == pytest.approx(np.array([0.00044044, 0.00044044]), 1.0e-4)


def test__coord_function_m__from():
    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=2.0
    )

    coord_m = mp.coord_func_m(grid_radius=np.array([2.0, 2.0]))

    assert coord_m == pytest.approx(np.array([0.0398826, 0.0398826]), 1.0e-4)

    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
    )

    coord_m = mp.coord_func_m(grid_radius=np.array([2.0, 2.0]))

    assert coord_m == pytest.approx(np.array([0.06726646, 0.06726646]), 1.0e-4)

    mp = ag.mp.NFWTruncatedSph(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=10.0, truncation_radius=3.0
    )

    coord_m = mp.coord_func_m(grid_radius=np.array([3.0, 3.0]))

    assert coord_m == pytest.approx(np.array([0.06946888, 0.06946888]), 1.0e-4)


def test__rho_at_scale_radius__unit_conversions():
    cosmology = ag.m.MockCosmology(
        arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0
    )

    mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    # When converting to kpc, the critical convergence is divided by kpc_per_arcsec**2.0 = 2.0**2.0
    # The scale radius also becomes scale_radius*kpc_per_arcsec = 2.0

    rho = mp.rho_at_scale_radius_solar_mass_per_kpc3(
        redshift_object=0.5, redshift_source=1.0, cosmology=cosmology
    )
    assert rho == pytest.approx(0.5 / 2.0, 1e-3)

    cosmology = ag.m.MockCosmology(
        arcsec_per_kpc=0.25, kpc_per_arcsec=4.0, critical_surface_density=2.0
    )

    rho = mp.rho_at_scale_radius_solar_mass_per_kpc3(
        redshift_object=0.5, redshift_source=1.0, cosmology=cosmology
    )
    assert rho == pytest.approx(0.5 / 4.0, 1e-3)

    cosmology = ag.m.MockCosmology(
        arcsec_per_kpc=0.25, kpc_per_arcsec=4.0, critical_surface_density=4.0
    )

    rho = mp.rho_at_scale_radius_solar_mass_per_kpc3(
        redshift_object=0.5, redshift_source=1.0, cosmology=cosmology
    )
    assert rho == pytest.approx(0.25 / 4.0, 1e-3)


def test__delta_concentration_value_in_default_units():
    mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    cosmology = ag.m.MockCosmology(
        arcsec_per_kpc=1.0,
        kpc_per_arcsec=1.0,
        critical_surface_density=1.0,
        cosmic_average_density=1.0,
    )

    delta_concentration = mp.delta_concentration(
        redshift_object=0.5, redshift_source=1.0, cosmology=cosmology
    )
    assert delta_concentration == pytest.approx(1.0, 1e-3)

    mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0)
    delta_concentration = mp.delta_concentration(
        redshift_object=0.5, redshift_source=1.0, cosmology=cosmology
    )
    assert delta_concentration == pytest.approx(3.0, 1e-3)

    mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)
    delta_concentration = mp.delta_concentration(
        redshift_object=0.5, redshift_source=1.0, cosmology=cosmology
    )
    assert delta_concentration == pytest.approx(0.25, 1e-3)


def test__solve_concentration():
    cosmology = ag.m.MockCosmology(
        arcsec_per_kpc=1.0,
        kpc_per_arcsec=1.0,
        critical_surface_density=1.0,
        cosmic_average_density=1.0,
    )

    mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    concentration = mp.concentration(
        redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology
    )

    assert concentration == pytest.approx(0.0074263, 1.0e-4)


def test__radius_at_200__different_length_units_include_conversions():
    mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    cosmology = ag.m.MockCosmology(arcsec_per_kpc=0.2, kpc_per_arcsec=5.0)

    concentration = mp.concentration(
        cosmology=cosmology, redshift_profile=0.5, redshift_source=1.0
    )

    radius_200 = mp.radius_at_200(
        redshift_object=0.5, redshift_source=1.0, cosmology=cosmology
    )

    assert radius_200 == concentration * 1.0


def test__mass_at_200__unit_conversions_work():
    mp = ag.mp.NFWSph(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

    cosmology = ag.m.MockCosmology(
        arcsec_per_kpc=1.0,
        kpc_per_arcsec=1.0,
        critical_surface_density=1.0,
        cosmic_average_density=1.0,
    )

    radius_at_200 = mp.radius_at_200(
        redshift_object=0.5, redshift_source=1.0, cosmology=cosmology
    )

    mass_at_200 = mp.mass_at_200_solar_masses(
        cosmology=cosmology, redshift_object=0.5, redshift_source=1.0
    )

    mass_calc = (
        200.0
        * ((4.0 / 3.0) * np.pi)
        * cosmology.cosmic_average_density
        * (radius_at_200**3.0)
    )
    assert mass_at_200 == pytest.approx(mass_calc, 1.0e-5)

    # cosmology = ag.m.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=2.0,
    #                           cosmic_average_density=1.0)
    #
    # radius_at_200 = mp.radius_at_200_for_units(unit_length='arcsec', redshift_galaxy=0.5, redshift_source=1.0,
    #                                             cosmology=cosmology)
    #
    # mass_at_200 = mp.mass_at_200(cosmology=cosmology, redshift_galaxy=0.5, redshift_source=1.0, unit_length='arcsec',
    #                               unit_mass='solMass')
    #
    # mass_calc = 200.0 * ((4.0 / 3.0) * np.pi) * cosmology.cosmic_average_density * (radius_at_200 ** 3.0)
    # assert mass_at_200 == pytest.approx(mass_calc, 1.0e-5)


def test__values_of_quantities_for_real_cosmology():
    cosmology = ag.cosmo.LambdaCDM(H0=70.0, Om0=0.3, Ode0=0.7)

    mp = ag.mp.NFWTruncatedSph(kappa_s=0.5, scale_radius=5.0, truncation_radius=10.0)

    rho = mp.rho_at_scale_radius_solar_mass_per_kpc3(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )

    delta_concentration = mp.delta_concentration(
        redshift_object=0.6,
        redshift_source=2.5,
        redshift_of_cosmic_average_density="local",
        cosmology=cosmology,
    )

    concentration = mp.concentration(
        redshift_profile=0.6,
        redshift_source=2.5,
        redshift_of_cosmic_average_density="local",
        cosmology=cosmology,
    )

    radius_at_200 = mp.radius_at_200(
        redshift_object=0.6,
        redshift_source=2.5,
        redshift_of_cosmic_average_density="local",
        cosmology=cosmology,
    )

    mass_at_200 = mp.mass_at_200_solar_masses(
        redshift_object=0.6,
        redshift_source=2.5,
        redshift_of_cosmic_average_density="local",
        cosmology=cosmology,
    )

    mass_at_truncation_radius = mp.mass_at_truncation_radius_solar_mass(
        redshift_profile=0.6,
        redshift_source=2.5,
        redshift_of_cosmic_average_density="local",
        cosmology=cosmology,
    )

    assert rho == pytest.approx(29027857.01622403, 1.0e-4)
    assert delta_concentration == pytest.approx(213451.19421263796, 1.0e-4)
    assert concentration == pytest.approx(18.6605624462417, 1.0e-4)
    assert radius_at_200 == pytest.approx(93.302812, 1.0e-4)
    assert mass_at_200 == pytest.approx(27651532986258.375, 1.0e-4)
    assert mass_at_truncation_radius == pytest.approx(14877085957074.299, 1.0e-4)

    rho = mp.rho_at_scale_radius_solar_mass_per_kpc3(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )

    delta_concentration = mp.delta_concentration(
        redshift_object=0.6,
        redshift_source=2.5,
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmology,
    )

    concentration = mp.concentration(
        redshift_profile=0.6,
        redshift_source=2.5,
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmology,
    )

    radius_at_200 = mp.radius_at_200(
        redshift_object=0.6,
        redshift_source=2.5,
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmology,
    )

    mass_at_200 = mp.mass_at_200_solar_masses(
        redshift_object=0.6,
        redshift_source=2.5,
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmology,
    )

    mass_at_truncation_radius = mp.mass_at_truncation_radius_solar_mass(
        redshift_profile=0.6,
        redshift_source=2.5,
        redshift_of_cosmic_average_density="profile",
        cosmology=cosmology,
    )

    assert rho == pytest.approx(29027857.01622403, 1.0e-4)
    assert delta_concentration == pytest.approx(110665.28111397651, 1.0e-4)
    assert concentration == pytest.approx(14.401574489517804, 1.0e-4)
    assert radius_at_200 == pytest.approx(72.007872, 1.0e-4)
    assert mass_at_200 == pytest.approx(24516707575366.09, 1.0e-4)
    assert mass_at_truncation_radius == pytest.approx(13190486262169.797, 1.0e-4)
