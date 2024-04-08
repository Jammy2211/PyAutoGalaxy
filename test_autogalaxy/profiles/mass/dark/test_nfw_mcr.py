import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__mass_and_concentration_consistent_with_normal_nfw():
    cosmology = ag.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.NFWMCRDuffySph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.NFWSph(
        centre=(1.0, 2.0),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert isinstance(mp.kappa_s, float)

    assert mp.centre == (1.0, 2.0)

    assert mp.axis_ratio == 1.0
    assert isinstance(mp.axis_ratio, float)

    assert mp.angle == 0.0
    assert isinstance(mp.angle, float)

    assert mp.inner_slope == 1.0
    assert isinstance(mp.inner_slope, float)

    assert mp.scale_radius == pytest.approx(0.273382, 1.0e-4)


def test__mass_and_concentration_consistent_with_normal_nfw__scatter_0():
    cosmology = ag.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.NFWMCRLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.NFWSph(
        centre=(1.0, 2.0),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert isinstance(mp.kappa_s, float)

    assert mp.centre == (1.0, 2.0)

    assert mp.axis_ratio == 1.0
    assert isinstance(mp.axis_ratio, float)

    assert mp.angle == 0.0
    assert isinstance(mp.angle, float)

    assert mp.inner_slope == 1.0
    assert isinstance(mp.inner_slope, float)

    assert mp.scale_radius == pytest.approx(0.21157, 1.0e-4)

    deflections_ludlow = mp.deflections_yx_2d_from(grid=grid)
    deflections = nfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()


def test__same_as_above_but_elliptical():
    cosmology = ag.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.NFWMCRLudlow(
        centre=(1.0, 2.0),
        ell_comps=(0.1, 0.2),
        mass_at_200=1.0e9,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.NFW(
        centre=(1.0, 2.0),
        ell_comps=(0.1, 0.2),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert isinstance(mp.kappa_s, float)

    assert mp.centre == (1.0, 2.0)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.1, 0.2))

    assert mp.axis_ratio == axis_ratio
    assert isinstance(mp.axis_ratio, float)

    assert mp.angle == angle
    assert isinstance(mp.angle, float)

    assert mp.inner_slope == 1.0
    assert isinstance(mp.inner_slope, float)

    assert mp.scale_radius == pytest.approx(0.211578, 1.0e-4)

    deflections_ludlow = mp.deflections_yx_2d_from(grid=grid)
    deflections = nfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()


def test__same_as_above_but_generalized_elliptical():
    cosmology = ag.cosmo.FlatLambdaCDM(H0=70.0, Om0=0.3)

    mp = ag.mp.gNFWMCRLudlow(
        centre=(1.0, 2.0),
        ell_comps=(0.1, 0.2),
        mass_at_200=1.0e9,
        inner_slope=2.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    mass_at_200_via_mass = mp.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_mass = mp.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    nfw_kappa_s = ag.mp.gNFW(
        centre=(1.0, 2.0),
        ell_comps=(0.1, 0.2),
        kappa_s=mp.kappa_s,
        scale_radius=mp.scale_radius,
        inner_slope=2.0,
    )

    mass_at_200_via_kappa_s = nfw_kappa_s.mass_at_200_solar_masses(
        redshift_object=0.6, redshift_source=2.5, cosmology=cosmology
    )
    concentration_via_kappa_s = nfw_kappa_s.concentration(
        redshift_profile=0.6, redshift_source=2.5, cosmology=cosmology
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mass_at_200_via_kappa_s == mass_at_200_via_mass
    assert concentration_via_kappa_s == concentration_via_mass

    assert isinstance(mp.kappa_s, float)

    assert mp.centre == (1.0, 2.0)

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=(0.1, 0.2))

    assert mp.axis_ratio == axis_ratio
    assert isinstance(mp.axis_ratio, float)

    assert mp.angle == angle
    assert isinstance(mp.angle, float)

    assert mp.inner_slope == 2.0
    assert isinstance(mp.inner_slope, float)

    assert mp.scale_radius == pytest.approx(0.21157, 1.0e-4)

    deflections_ludlow = mp.deflections_yx_2d_from(grid=grid)
    deflections = nfw_kappa_s.deflections_yx_2d_from(grid=grid)

    assert (deflections_ludlow == deflections).all()
