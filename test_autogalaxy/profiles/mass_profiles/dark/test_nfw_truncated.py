import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_yx_2d_from():

    truncated_nfw = ag.mp.SphNFWTruncated(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=2.0
    )

    # factor = (4.0 * kappa_s * scale_radius / (r / scale_radius))

    deflections = truncated_nfw.deflections_yx_2d_from(grid=np.array([[2.0, 0.0]]))

    factor = (4.0 * 1.0 * 1.0) / (2.0 / 1.0)
    assert deflections[0, 0] == pytest.approx(factor * 0.38209715, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

    deflections = truncated_nfw.deflections_yx_2d_from(grid=np.array([[0.0, 2.0]]))

    assert deflections[0, 0] == pytest.approx(0.0, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(factor * 0.38209715, 1.0e-4)

    deflections = truncated_nfw.deflections_yx_2d_from(grid=np.array([[1.0, 1.0]]))

    factor = (4.0 * 1.0 * 1.0) / (np.sqrt(2) / 1.0)
    assert deflections[0, 0] == pytest.approx(
        (1.0 / np.sqrt(2)) * factor * 0.3125838, 1.0e-4
    )
    assert deflections[0, 1] == pytest.approx(
        (1.0 / np.sqrt(2)) * factor * 0.3125838, 1.0e-4
    )

    truncated_nfw = ag.mp.SphNFWTruncated(
        centre=(0.0, 0.0), kappa_s=2.0, scale_radius=1.0, truncation_radius=2.0
    )

    deflections = truncated_nfw.deflections_yx_2d_from(grid=np.array([[2.0, 0.0]]))

    factor = (4.0 * 2.0 * 1.0) / (2.0 / 1.0)
    assert deflections[0, 0] == pytest.approx(factor * 0.38209715, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

    truncated_nfw = ag.mp.SphNFWTruncated(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0, truncation_radius=2.0
    )

    deflections = truncated_nfw.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([(2.0, 0.0)])
    )

    assert deflections[0, 0] == pytest.approx(2.1702661386, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)


def test__convergence_2d_from():

    truncated_nfw = ag.mp.SphNFWTruncated(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=2.0
    )

    convergence = truncated_nfw.convergence_2d_from(grid=np.array([[2.0, 0.0]]))

    assert convergence == pytest.approx(2.0 * 0.046409642, 1.0e-4)

    convergence = truncated_nfw.convergence_2d_from(grid=np.array([[1.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 0.10549515, 1.0e-4)

    truncated_nfw = ag.mp.SphNFWTruncated(
        centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0, truncation_radius=2.0
    )

    convergence = truncated_nfw.convergence_2d_from(grid=np.array([[2.0, 0.0]]))

    assert convergence == pytest.approx(6.0 * 0.046409642, 1.0e-4)

    truncated_nfw = ag.mp.SphNFWTruncated(
        centre=(0.0, 0.0), kappa_s=3.0, scale_radius=5.0, truncation_radius=2.0
    )

    convergence = truncated_nfw.convergence_2d_from(grid=np.array([[2.0, 0.0]]))

    assert convergence == pytest.approx(1.51047026, 1.0e-4)


def test__mass_at_truncation_radius():

    truncated_nfw = ag.mp.SphNFWTruncated(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0, truncation_radius=1.0
    )

    cosmology = ag.m.MockCosmology(
        arcsec_per_kpc=1.0,
        kpc_per_arcsec=1.0,
        critical_surface_density=1.0,
        cosmic_average_density=1.0,
    )

    mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius_solar_mass(
        redshift_profile=0.5, redshift_source=1.0, cosmology=cosmology
    )

    assert mass_at_truncation_radius == pytest.approx(0.00009792581, 1.0e-5)

    # truncated_nfw = ag.mp.SphNFWTruncated(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0,
    #                                          truncation_radius=1.0)
    #
    # cosmology = ag.m.MockCosmology(arcsec_per_kpc=1.0, kpc_per_arcsec=1.0, critical_surface_density=2.0,
    #                           cosmic_average_density=3.0)
    #
    # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_galaxy=0.5, redshift_source=1.0,
    #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
    #
    # assert mass_at_truncation_radius == pytest.approx(0.00008789978, 1.0e-5)
    #
    # truncated_nfw = ag.mp.SphNFWTruncated(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=2.0,
    #                                          truncation_radius=1.0)
    #
    # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_galaxy=0.5, redshift_source=1.0,
    #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
    #
    # assert mass_at_truncation_radius == pytest.approx(0.0000418378, 1.0e-5)
    #
    # truncated_nfw = ag.mp.SphNFWTruncated(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=8.0,
    #                                          truncation_radius=4.0)
    #
    # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_galaxy=0.5, redshift_source=1.0,
    #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
    #
    # assert mass_at_truncation_radius == pytest.approx(0.0000421512, 1.0e-4)
    #
    # truncated_nfw = ag.mp.SphNFWTruncated(centre=(0.0, 0.0), kappa_s=2.0, scale_radius=8.0,
    #                                          truncation_radius=4.0)
    #
    # mass_at_truncation_radius = truncated_nfw.mass_at_truncation_radius(redshift_galaxy=0.5, redshift_source=1.0,
    #     unit_length='arcsec', unit_mass='solMass', cosmology=cosmology)
    #
    # assert mass_at_truncation_radius == pytest.approx(0.00033636625, 1.0e-4)


def test__compare_nfw_and_truncated_nfw_with_large_truncation_radius():

    truncated_nfw = ag.mp.SphNFWTruncated(
        centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0, truncation_radius=50000.0
    )

    nfw = ag.mp.SphNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=4.0)

    truncated_nfw_convergence = truncated_nfw.convergence_2d_from(
        grid=np.array([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
    )
    nfw_convergence = nfw.convergence_2d_from(
        grid=np.array([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
    )

    assert truncated_nfw_convergence == pytest.approx(nfw_convergence, 1.0e-4)

    truncated_nfw_deflections = truncated_nfw.deflections_yx_2d_from(
        grid=np.array([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
    )
    nfw_deflections = nfw.deflections_yx_2d_from(
        grid=np.array([[2.0, 2.0], [3.0, 1.0], [-1.0, -9.0]])
    )

    assert truncated_nfw_deflections == pytest.approx(nfw_deflections, 1.0e-4)