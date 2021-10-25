import pytest

import autogalaxy as ag
from autogalaxy.profiles.geometry_profiles import GeometryProfile


@pytest.fixture(name="ell_sersic")
def make_ell_sersic():
    return ag.mp.EllSersic()


@pytest.fixture(name="ell_sersic_dict")
def make_ell_sersic_dict():
    return {
        "type": "autogalaxy.profiles.mass_profiles.stellar_mass_profiles.EllSersic",
        "centre": (0.0, 0.0),
        "elliptical_comps": (0.0, 0.0),
        "intensity": 0.1,
        "effective_radius": 0.6,
        "sersic_index": 0.6,
        "mass_to_light_ratio": 1.0,
    }


def test_to_dict(ell_sersic, ell_sersic_dict):
    assert ell_sersic.dict() == ell_sersic_dict


def test_from_dict(ell_sersic, ell_sersic_dict):
    assert ell_sersic == GeometryProfile.from_dict(ell_sersic_dict)
