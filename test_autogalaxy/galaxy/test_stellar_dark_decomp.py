import autogalaxy as ag

from autogalaxy import exc

import pytest


def test__stellar_mass_angular_within_galaxy__is_sum_of_individual_profiles(
    smp_0, smp_1
):

    galaxy = ag.Galaxy(
        redshift=0.5,
        stellar_0=smp_0,
        non_stellar_profile=ag.mp.EllIsothermal(einstein_radius=1.0),
    )
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    stellar_mass_0 = smp_0.mass_angular_within_circle_from(radius=0.5)

    gal_mass = decomp.stellar_mass_angular_within_circle_from(radius=0.5)

    assert stellar_mass_0 == gal_mass

    galaxy = ag.Galaxy(
        redshift=0.5,
        stellar_0=smp_0,
        stellar_1=smp_1,
        non_stellar_profile=ag.mp.EllIsothermal(einstein_radius=1.0),
    )
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    stellar_mass_1 = smp_1.mass_angular_within_circle_from(radius=0.5)

    gal_mass = decomp.stellar_mass_angular_within_circle_from(radius=0.5)

    assert stellar_mass_0 + stellar_mass_1 == gal_mass

    galaxy = ag.Galaxy(redshift=0.5)
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    with pytest.raises(exc.GalaxyException):
        decomp.stellar_mass_angular_within_circle_from(radius=1.0)


def test__stellar_fraction_at_radius(dmp_0, dmp_1, smp_0, smp_1):

    galaxy = ag.Galaxy(redshift=0.5, stellar_0=smp_0, dark_0=dmp_0)
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    stellar_mass_0 = smp_0.mass_angular_within_circle_from(radius=1.0)
    dark_mass_0 = dmp_0.mass_angular_within_circle_from(radius=1.0)

    stellar_fraction = decomp.stellar_fraction_at_radius_from(radius=1.0)

    assert stellar_fraction == pytest.approx(
        stellar_mass_0 / (dark_mass_0 + stellar_mass_0), 1.0e-4
    )

    galaxy = ag.Galaxy(redshift=0.5, stellar_0=smp_0, stellar_1=smp_1, dark_0=dmp_0)
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    stellar_fraction = decomp.stellar_fraction_at_radius_from(radius=1.0)
    stellar_mass_1 = smp_1.mass_angular_within_circle_from(radius=1.0)

    assert stellar_fraction == pytest.approx(
        (stellar_mass_0 + stellar_mass_1)
        / (dark_mass_0 + stellar_mass_0 + stellar_mass_1),
        1.0e-4,
    )

    galaxy = ag.Galaxy(
        redshift=0.5, stellar_0=smp_0, stellar_1=smp_1, dark_0=dmp_0, dark_mass_1=dmp_1
    )
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    stellar_fraction = decomp.stellar_fraction_at_radius_from(radius=1.0)
    dark_mass_1 = dmp_1.mass_angular_within_circle_from(radius=1.0)

    assert stellar_fraction == pytest.approx(
        (stellar_mass_0 + stellar_mass_1)
        / (dark_mass_0 + dark_mass_1 + stellar_mass_0 + stellar_mass_1),
        1.0e-4,
    )


def test__dark_mass_within_galaxy__is_sum_of_individual_profiles(dmp_0, dmp_1):

    galaxy = ag.Galaxy(
        redshift=0.5,
        dark_0=dmp_0,
        non_dark_profile=ag.mp.EllIsothermal(einstein_radius=1.0),
    )
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    dark_mass_0 = dmp_0.mass_angular_within_circle_from(radius=0.5)

    gal_mass = decomp.dark_mass_angular_within_circle_from(radius=0.5)

    assert dark_mass_0 == gal_mass

    galaxy = ag.Galaxy(
        redshift=0.5,
        dark_0=dmp_0,
        dark_1=dmp_1,
        non_dark_profile=ag.mp.EllIsothermal(einstein_radius=1.0),
    )
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    dark_mass_1 = dmp_1.mass_angular_within_circle_from(radius=0.5)

    gal_mass = decomp.dark_mass_angular_within_circle_from(radius=0.5)

    assert dark_mass_0 + dark_mass_1 == gal_mass

    galaxy = ag.Galaxy(redshift=0.5)
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    with pytest.raises(exc.GalaxyException):
        decomp.dark_mass_angular_within_circle_from(radius=1.0)


def test__dark_fraction_at_radius(dmp_0, dmp_1, smp_0, smp_1):

    galaxy = ag.Galaxy(redshift=0.5, dark_0=dmp_0, stellar_0=smp_0)
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    stellar_mass_0 = smp_0.mass_angular_within_circle_from(radius=1.0)
    dark_mass_0 = dmp_0.mass_angular_within_circle_from(radius=1.0)

    dark_fraction = decomp.dark_fraction_at_radius_from(radius=1.0)

    assert dark_fraction == dark_mass_0 / (stellar_mass_0 + dark_mass_0)

    galaxy = ag.Galaxy(redshift=0.5, dark_0=dmp_0, dark_1=dmp_1, stellar_0=smp_0)
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    dark_fraction = decomp.dark_fraction_at_radius_from(radius=1.0)
    dark_mass_1 = dmp_1.mass_angular_within_circle_from(radius=1.0)

    assert dark_fraction == pytest.approx(
        (dark_mass_0 + dark_mass_1) / (stellar_mass_0 + dark_mass_0 + dark_mass_1),
        1.0e-4,
    )

    galaxy = ag.Galaxy(
        redshift=0.5, dark_0=dmp_0, dark_1=dmp_1, stellar_0=smp_0, stellar_mass_1=smp_1
    )
    decomp = ag.StellarDarkDecomp(galaxy=galaxy)

    dark_fraction = decomp.dark_fraction_at_radius_from(radius=1.0)
    stellar_mass_1 = smp_1.mass_angular_within_circle_from(radius=1.0)

    assert dark_fraction == pytest.approx(
        (dark_mass_0 + dark_mass_1)
        / (stellar_mass_0 + stellar_mass_1 + dark_mass_0 + dark_mass_1),
        1.0e-4,
    )
