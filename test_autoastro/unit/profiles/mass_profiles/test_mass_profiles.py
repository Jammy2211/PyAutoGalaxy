import autofit as af
import autoarray as aa
import autoastro as aast
import math
import numpy as np
import pytest
import os

from test_autoastro.mock import mock_cosmology


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    af.conf.instance = af.conf.default


def mass_within_radius_of_profile_from_grid_calculation(radius, profile):

    mass_total = 0.0

    xs = np.linspace(-radius * 1.5, radius * 1.5, 40)
    ys = np.linspace(-radius * 1.5, radius * 1.5, 40)

    edge = xs[1] - xs[0]
    area = edge ** 2

    for x in xs:
        for y in ys:

            eta = profile.grid_to_elliptical_radii(
                grid=aa.grid_irregular.manual_1d([[x, y]])
            )

            if eta < radius:
                mass_total += profile.convergence_func(eta) * area

    return mass_total


# class TestMassWithinCircle(object):
#     def test__mass_in_angular_units__singular_isothermal_sphere__compare_to_analytic(
#         self
#     ):
#
#         sis = aast.mp.SphericalIsothermal(einstein_radius=2.0)
#
#         radius = aast.dim.Length(2.0, "arcsec")
#
#         mass = sis.mass_within_circle_in_units(
#             radius=radius, redshift_object=0.5, redshift_source=1.0, unit_mass="angular"
#         )
#         assert math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)
#
#         sis = aast.mp.SphericalIsothermal(einstein_radius=4.0)
#
#         radius = aast.dim.Length(4.0, "arcsec")
#
#         mass = sis.mass_within_circle_in_units(
#             radius=radius, redshift_object=0.5, redshift_source=1.0, unit_mass="angular"
#         )
#         assert math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)
#
#     def test__mass_in_angular_units__singular_isothermal__compare_to_grid(self):
#
#         sis = aast.mp.SphericalIsothermal(einstein_radius=2.0)
#
#         radius = aast.dim.Length(1.0, "arcsec")
#
#         mass_grid = mass_within_radius_of_profile_from_grid_calculation(
#             radius=radius, profile=sis
#         )
#
#         mass = sis.mass_within_circle_in_units(
#             radius=radius, redshift_object=0.5, redshift_source=1.0, unit_mass="angular"
#         )
#
#         assert mass_grid == pytest.approx(mass, 0.02)
#
#     def test__radius_units_conversions__mass_profile_updates_units_and_computes_correct_mass(
#         self
#     ):
#
#         cosmology = mock_cosmology.MockCosmology(kpc_per_arcsec=2.0)
#
#         # arcsec -> arcsec
#
#         sis_arcsec = aast.mp.SphericalIsothermal(
#             centre=(aast.dim.Length(0.0, "arcsec"), aast.dim.Length(0.0, "arcsec")),
#             einstein_radius=aast.dim.Length(2.0, "arcsec"),
#         )
#
#         radius = aast.dim.Length(2.0, "arcsec")
#         mass = sis_arcsec.mass_within_circle_in_units(
#             radius=radius, redshift_object=0.5, redshift_source=1.0, unit_mass="angular"
#         )
#         assert math.pi * sis_arcsec.einstein_radius * radius == pytest.approx(
#             mass, 1e-3
#         )
#
#         # arcsec -> kpc
#
#         radius = aast.dim.Length(2.0, "kpc")
#         mass = sis_arcsec.mass_within_circle_in_units(
#             radius=radius,
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="angular",
#             cosmology=cosmology,
#         )
#         assert math.pi * sis_arcsec.einstein_radius * 1.0 == pytest.approx(mass, 1e-3)
#
#         # 2.0 arcsec = 4.0 kpc, same masses.
#
#         radius = aast.dim.Length(2.0, "arcsec")
#         mass_arcsec = sis_arcsec.mass_within_circle_in_units(
#             radius=radius,
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="angular",
#             cosmology=cosmology,
#         )
#         radius = aast.dim.Length(4.0, "kpc")
#         mass_kpc = sis_arcsec.mass_within_circle_in_units(
#             radius=radius,
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="angular",
#             cosmology=cosmology,
#         )
#         assert mass_arcsec == mass_kpc
#
#         # kpc -> kpc
#
#         sis_kpc = aast.mp.SphericalIsothermal(
#             centre=(aast.dim.Length(0.0, "kpc"), aast.dim.Length(0.0, "kpc")),
#             einstein_radius=aast.dim.Length(2.0, "kpc"),
#         )
#
#         sis_kpc.core_radius = aast.dim.Length(value=0.0, unit_length="kpc")
#
#         radius = aast.dim.Length(2.0, "kpc")
#         mass = sis_kpc.mass_within_circle_in_units(
#             radius=radius,
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="angular",
#             cosmology=cosmology,
#         )
#         assert math.pi * sis_kpc.einstein_radius * radius == pytest.approx(mass, 1e-3)
#
#         # kpc -> arcsec
#
#         radius = aast.dim.Length(2.0, "arcsec")
#         mass = sis_kpc.mass_within_circle_in_units(
#             radius=radius,
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="angular",
#             cosmology=cosmology,
#         )
#         assert 2.0 * math.pi * sis_kpc.einstein_radius * radius == pytest.approx(
#             mass, 1e-3
#         )
#
#         # 2.0 arcsec = 4.0 kpc, same masses.
#
#         radius = aast.dim.Length(2.0, "arcsec")
#         mass_arcsec = sis_kpc.mass_within_circle_in_units(
#             radius=radius,
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="angular",
#             cosmology=cosmology,
#         )
#         radius = aast.dim.Length(4.0, "kpc")
#         mass_kpc = sis_kpc.mass_within_circle_in_units(
#             radius=radius, redshift_object=0.5, redshift_source=1.0, unit_mass="angular"
#         )
#         assert mass_arcsec == mass_kpc
#
#     def test__mass_units_conversions__multiplies_by_critical_surface_density_factor(
#         self
#     ):
#
#         cosmology = mock_cosmology.MockCosmology(critical_surface_density=2.0)
#
#         sis = aast.mp.SphericalIsothermal(einstein_radius=2.0)
#         radius = aast.dim.Length(2.0, "arcsec")
#
#         mass = sis.mass_within_circle_in_units(
#             radius=radius,
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="angular",
#             cosmology=cosmology,
#         )
#         assert math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)
#
#         mass = sis.mass_within_circle_in_units(
#             radius=radius,
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="solMass",
#             cosmology=cosmology,
#         )
#         assert 2.0 * math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)
#
#         mass = sis.mass_within_circle_in_units(
#             radius=radius,
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="solMass",
#             cosmology=cosmology,
#         )
#         assert 2.0 * math.pi * sis.einstein_radius * radius == pytest.approx(mass, 1e-3)
#
#
# class TestDensityBetweenAnnuli(object):
#     def test__circular_annuli__sis__analyic_density_agrees(self):
#
#         cosmology = mock_cosmology.MockCosmology(
#             kpc_per_arcsec=2.0, critical_surface_density=2.0
#         )
#
#         einstein_radius = 1.0
#         sis_arcsec = aast.mp.SphericalIsothermal(
#             centre=(0.0, 0.0), einstein_radius=einstein_radius
#         )
#
#         inner_annuli_radius = aast.dim.Length(2.0, "arcsec")
#         outer_annuli_radius = aast.dim.Length(3.0, "arcsec")
#
#         inner_mass = math.pi * einstein_radius * inner_annuli_radius
#         outer_mass = math.pi * einstein_radius * outer_annuli_radius
#
#         density_between_annuli = sis_arcsec.density_between_circular_annuli_in_angular_units(
#             inner_annuli_radius=inner_annuli_radius,
#             outer_annuli_radius=outer_annuli_radius,
#             unit_length="arcsec",
#             unit_mass="angular",
#             redshift_profile=0.5,
#             redshift_source=1.0,
#             cosmology=cosmology,
#         )
#
#         annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (
#             np.pi * inner_annuli_radius ** 2.0
#         )
#
#         assert (outer_mass - inner_mass) / annuli_area == pytest.approx(
#             density_between_annuli, 1e-4
#         )
#
#         density_between_annuli = sis_arcsec.density_between_circular_annuli_in_angular_units(
#             inner_annuli_radius=inner_annuli_radius,
#             outer_annuli_radius=outer_annuli_radius,
#             unit_length="arcsec",
#             unit_mass="solMass",
#             redshift_profile=0.5,
#             redshift_source=1.0,
#             cosmology=cosmology,
#         )
#
#         annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (
#             np.pi * inner_annuli_radius ** 2.0
#         )
#
#         assert (2.0 * outer_mass - 2.0 * inner_mass) / annuli_area == pytest.approx(
#             density_between_annuli, 1e-4
#         )
#
#         density_between_annuli = sis_arcsec.density_between_circular_annuli_in_angular_units(
#             inner_annuli_radius=inner_annuli_radius,
#             outer_annuli_radius=outer_annuli_radius,
#             unit_length="kpc",
#             unit_mass="angular",
#             redshift_profile=0.5,
#             redshift_source=1.0,
#             cosmology=cosmology,
#         )
#
#         inner_mass = math.pi * 2.0 * einstein_radius * inner_annuli_radius
#         outer_mass = math.pi * 2.0 * einstein_radius * outer_annuli_radius
#
#         annuli_area = (np.pi * 2.0 * outer_annuli_radius ** 2.0) - (
#             np.pi * 2.0 * inner_annuli_radius ** 2.0
#         )
#
#         assert (outer_mass - inner_mass) / annuli_area == pytest.approx(
#             density_between_annuli, 1e-4
#         )
#
#     def test__circular_annuli__nfw_profile__compare_to_manual_mass(self):
#
#         cosmology = mock_cosmology.MockCosmology(
#             kpc_per_arcsec=2.0, critical_surface_density=2.0
#         )
#
#         nfw = aast.mp.EllipticalNFW(
#             centre=(0.0, 0.0), axis_ratio=0.8, phi=45.0, kappa_s=1.0
#         )
#
#         inner_mass = nfw.mass_within_circle_in_units(
#             radius=aast.dim.Length(1.0),
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="angular",
#         )
#
#         outer_mass = nfw.mass_within_circle_in_units(
#             radius=aast.dim.Length(2.0),
#             redshift_object=0.5,
#             redshift_source=1.0,
#             unit_mass="angular",
#         )
#
#         density_between_annuli = nfw.density_between_circular_annuli_in_angular_units(
#             inner_annuli_radius=aast.dim.Length(1.0),
#             outer_annuli_radius=aast.dim.Length(2.0),
#             unit_length="arcsec",
#             unit_mass="angular",
#             redshift_profile=0.5,
#             redshift_source=1.0,
#             cosmology=cosmology,
#         )
#
#         annuli_area = (np.pi * 2.0 ** 2.0) - (np.pi * 1.0 ** 2.0)
#
#         assert (outer_mass - inner_mass) / annuli_area == pytest.approx(
#             density_between_annuli, 1e-4
#         )
#
#
# class TestLensingObject:
#     def test__mass_profiles__list__is_list_of_self(self):
#
#         sis = aast.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)
#
#         assert sis.mass_profiles == [sis]
#
#     def test__correct_einstein_mass_caclulated__means_all_innherited_methods_work(
#         self,
#     ):
#
#         sis = aast.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)
#
#         assert sis.einstein_mass_in_units(unit_mass="angular") == pytest.approx(
#             np.pi * 2.0 ** 2.0, 1.0e-2
#         )
