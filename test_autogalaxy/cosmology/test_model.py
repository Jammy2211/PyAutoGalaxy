import pytest

import autogalaxy as ag


def test__Planck15_Om0(Planck15):
    Planck15_Om0 = ag.cosmo.model.Planck15Om0()

    critical_surface_density = Planck15.critical_surface_density_between_redshifts_from(
        redshift_0=0.1, redshift_1=1.0
    )

    critical_surface_density_Om0 = (
        Planck15_Om0.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density - critical_surface_density_Om0 < 1e-4

    Planck15_Om0 = ag.cosmo.model.Planck15Om0(Om0=0.5)

    critical_surface_density = Planck15.critical_surface_density_between_redshifts_from(
        redshift_0=0.1, redshift_1=1.0
    )

    critical_surface_density_Om0 = (
        Planck15_Om0.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density - critical_surface_density_Om0 > 10.0


def test__Planck15_flat_w(Planck15):
    Planck15_flat_w = ag.cosmo.model.Planck15FlatwCDM()

    critical_surface_density = Planck15.critical_surface_density_between_redshifts_from(
        redshift_0=0.1, redshift_1=1.0
    )

    critical_surface_density_flat_w = (
        Planck15_flat_w.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density - critical_surface_density_flat_w < 1.0e-4

    Planck15_flat_w = ag.cosmo.model.Planck15FlatwCDM(Om0=0.1)

    critical_surface_density = Planck15.critical_surface_density_between_redshifts_from(
        redshift_0=0.1, redshift_1=1.0
    )

    critical_surface_density_flat_w = (
        Planck15_flat_w.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density - critical_surface_density_flat_w > 10.0

    Planck15_flat_w = ag.cosmo.model.Planck15FlatwCDM(Om0=0.1, w0=-0.5)

    critical_surface_density = Planck15.critical_surface_density_between_redshifts_from(
        redshift_0=0.1, redshift_1=1.0
    )

    critical_surface_density_flat_w = (
        Planck15_flat_w.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density - critical_surface_density_flat_w > 10.0
