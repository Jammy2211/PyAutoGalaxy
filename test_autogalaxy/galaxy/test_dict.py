import pytest

import autogalaxy as ag
from autoarray.inversion.pixelizations.voronoi import Voronoi
from autoarray.inversion.regularization import AdaptiveBrightness
from autoconf.dictable import Dictable


@pytest.fixture(name="trivial_galaxy")
def make_trivial_galaxy():
    return ag.Galaxy(redshift=1.0)


@pytest.fixture(name="trivial_galaxy_dict")
def make_trivial_galaxy_dict():
    return {
        "hyper_galaxy": None,
        "pixelization": None,
        "redshift": 1.0,
        "regularization": None,
        "type": "autogalaxy.galaxy.galaxy.Galaxy",
    }


class TestTrivial:
    def test_dict(self, trivial_galaxy, trivial_galaxy_dict):
        assert trivial_galaxy.dict() == trivial_galaxy_dict

    def test_from_dict(self, trivial_galaxy, trivial_galaxy_dict):
        assert Dictable.from_dict(trivial_galaxy_dict) == trivial_galaxy


@pytest.fixture(name="ization_galaxy")
def make_ization_galaxy():
    return ag.Galaxy(
        redshift=1.0, pixelization=Voronoi(), regularization=AdaptiveBrightness()
    )


@pytest.fixture(name="ization_galaxy_dict")
def make_ization_galaxy_dict():
    return {
        "hyper_galaxy": None,
        "pixelization": {"type": "autoarray.inversion.pixelizations.voronoi.Voronoi"},
        "redshift": 1.0,
        "regularization": {
            "inner_coefficient": 1.0,
            "outer_coefficient": 1.0,
            "signal_scale": 1.0,
            "type": "autoarray.inversion.regularization.adaptive_brightness.AdaptiveBrightness",
        },
        "type": "autogalaxy.galaxy.galaxy.Galaxy",
    }


class TestIzations:
    def test_dict(self, ization_galaxy, ization_galaxy_dict):
        assert ization_galaxy.dict() == ization_galaxy_dict

    def test_from_dict(self, ization_galaxy, ization_galaxy_dict):
        galaxy = Dictable.from_dict(ization_galaxy_dict)
        assert galaxy == ization_galaxy


def test_pixelization_equality():
    assert Voronoi() == Voronoi()


@pytest.fixture(name="profiles_galaxy")
def make_profiles_galaxy():
    return ag.Galaxy(
        redshift=2.0, light=ag.lp.EllChameleon(), mass=ag.mp.EllDevVaucouleurs()
    )


@pytest.fixture(name="profiles_galaxy_dict")
def make_profiles_galaxy_dict():
    return {
        "hyper_galaxy": None,
        "light": {
            "centre": (0.0, 0.0),
            "core_radius_0": 0.01,
            "core_radius_1": 0.05,
            "elliptical_comps": (0.0, 0.0),
            "intensity": 0.1,
            "type": "autogalaxy.profiles.light_profiles.light_profiles.EllChameleon",
        },
        "mass": {
            "centre": (0.0, 0.0),
            "effective_radius": 0.6,
            "elliptical_comps": (0.0, 0.0),
            "intensity": 0.1,
            "mass_to_light_ratio": 1.0,
            "type": "autogalaxy.profiles.mass_profiles.stellar_mass_profiles.EllDevVaucouleurs",
        },
        "pixelization": None,
        "redshift": 2.0,
        "regularization": None,
        "type": "autogalaxy.galaxy.galaxy.Galaxy",
    }


class TestProfiles:
    def test_dict(self, profiles_galaxy, profiles_galaxy_dict):
        assert profiles_galaxy.dict() == profiles_galaxy_dict

    def test_from_dict(self, profiles_galaxy, profiles_galaxy_dict):
        assert Dictable.from_dict(profiles_galaxy_dict) == profiles_galaxy
