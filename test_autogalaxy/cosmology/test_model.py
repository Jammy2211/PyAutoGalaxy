import pytest

import autogalaxy as ag


def test__cosmology(Planck15):
    cosmology = ag.cosmo.FlatwCDMWrap()

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density == pytest.approx(17613991217.945473, 1.0e-4)

    cosmology = ag.cosmo.FlatLambdaCDMWrap()

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density == pytest.approx(17613991217.945473, 1.0e-4)
