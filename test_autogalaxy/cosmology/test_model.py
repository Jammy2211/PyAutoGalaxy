import pytest

def test__cosmology(Planck15):

    from autogalaxy.cosmology.model import FlatwCDMWrap
    cosmology = FlatwCDMWrap()

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density == pytest.approx(17613991217.945473, 1.0e-4)

    from autogalaxy.cosmology.model import FlatLambdaCDMWrap
    cosmology = FlatLambdaCDMWrap()

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_from(
            redshift_0=0.1, redshift_1=1.0
        )
    )

    assert critical_surface_density == pytest.approx(17613991217.945473, 1.0e-4)
