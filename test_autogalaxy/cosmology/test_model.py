import pytest

from astropy import cosmology as cosmo

import autogalaxy as ag

def test__angular_diameter_distance():

    cosmology_ap = cosmo.FlatwCDM(H0=70, Om0=0.3, w0=-1.0, Tcmb0=2.725)

    angular_diameter_distance_ap = cosmology_ap.angular_diameter_distance_z1z2(0.1, 1.0).to("kpc").value

    cosmology = ag.cosmo.FlatwCDMWrap(H0=70, Om0=0.3, w0=-1.0, Tcmb0=2.725)

    angular_diameter_distance = (
        cosmology.angular_diameter_distance_kpc_z1z2(0.1, 1.0)
    )

    assert angular_diameter_distance == pytest.approx(angular_diameter_distance_ap, 1.0e-4)

def test__critical_surface_density():

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
