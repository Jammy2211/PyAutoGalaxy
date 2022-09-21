import pytest

import autogalaxy as ag
from autoarray.inversion.pixelization.mesh.voronoi import Voronoi
from autoarray.inversion.regularization import AdaptiveBrightness
from autoconf.dictable import Dictable


@pytest.fixture(name="trivial_galaxy")
def make_trivial_galaxy():
    return ag.Galaxy(redshift=1.0)


@pytest.fixture(name="trivial_galaxy_dict")
def make_trivial_galaxy_dict():
    return {
        "hyper_galaxy": None,
        "redshift": 1.0,
        "type": "autogalaxy.galaxy.galaxy.Galaxy",
    }


def test__trivial_dict(trivial_galaxy, trivial_galaxy_dict):
    assert trivial_galaxy.dict() == trivial_galaxy_dict


def test_trivial__from_dict(trivial_galaxy, trivial_galaxy_dict):
    assert Dictable.from_dict(trivial_galaxy_dict) == trivial_galaxy


@pytest.fixture(name="pixelization_galaxy")
def make_pixelization_galaxy():
    return ag.PixelizationGalaxy(
        redshift=1.0,
        pixelization=ag.Pixelization(
            mesh=Voronoi(), regularization=AdaptiveBrightness()
        ),
    )


@pytest.fixture(name="pixelization_galaxy_dict")
def make_pixelization_galaxy_dict():
    return {
        "hyper_galaxy": None,
        "pixelization": {
            "mesh": {"type": "autoarray.inversion.pixelization.mesh.voronoi.Voronoi"},
            "regularization": {
                "inner_coefficient": 1.0,
                "outer_coefficient": 1.0,
                "signal_scale": 1.0,
                "type": "autoarray.inversion.regularization.adaptive_brightness.AdaptiveBrightness",
            },
            "type": "autoarray.inversion.pixelization.pixelization.Pixelization",
        },
        "redshift": 1.0,
        "type": "autogalaxy.galaxy.galaxy.PixelizationGalaxy",
    }


# TODO : Rich fix these to work with Pixelization objects (e.g. contained of mesh and regularization)


def test__with_pixelization__dict(pixelization_galaxy, pixelization_galaxy_dict):
<<<<<<< HEAD
=======

    print(pixelization_galaxy_dict)

    print(pixelization_galaxy.dict())

>>>>>>> feature/inversion_docs
    assert pixelization_galaxy.dict() == pixelization_galaxy_dict


def test__with_pixelization__from_dict(pixelization_galaxy, pixelization_galaxy_dict):
    galaxy = Dictable.from_dict(pixelization_galaxy_dict)
    assert galaxy == pixelization_galaxy


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
        "redshift": 2.0,
        "type": "autogalaxy.galaxy.galaxy.Galaxy",
    }


def test__profiles_galaxy__dict(profiles_galaxy, profiles_galaxy_dict):
    assert profiles_galaxy.dict() == profiles_galaxy_dict


def test__profiles_galaxy__from_dict(profiles_galaxy, profiles_galaxy_dict):
    assert Dictable.from_dict(profiles_galaxy_dict) == profiles_galaxy
