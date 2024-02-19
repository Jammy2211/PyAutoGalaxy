import pytest
from typing import Optional

import autogalaxy as ag
from autoarray.inversion.pixelization.mesh.voronoi import Voronoi
from autoarray.inversion.regularization import AdaptiveBrightness
from autoconf.dictable import from_dict


@pytest.fixture(name="trivial_galaxy")
def make_trivial_galaxy():
    return ag.Galaxy(redshift=1.0)


@pytest.fixture(name="trivial_galaxy_dict")
def make_trivial_galaxy_dict():
    return {
        "type": "instance",
        "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
        "arguments": {"redshift": 1.0, "label": "cls345"},
    }


def test__trivial_dict(trivial_galaxy, trivial_galaxy_dict):
    assert trivial_galaxy.dict() == trivial_galaxy_dict


class PixelizationGalaxy(ag.Galaxy):
    def __init__(
        self, redshift: float, pixelization: Optional[ag.Pixelization] = None, **kwargs
    ):
        super().__init__(redshift, **kwargs)
        self.pixelization = pixelization


@pytest.fixture(name="pixelization_galaxy")
def make_pixelization_galaxy():
    return PixelizationGalaxy(
        redshift=1.0,
        pixelization=ag.Pixelization(
            image_mesh=ag.image_mesh.Overlay(shape=(3, 3)),
            mesh=Voronoi(),
            regularization=AdaptiveBrightness(),
        ),
    )


@pytest.fixture(name="pixelization_galaxy_dict")
def make_pixelization_galaxy_dict():
    return {
        "type": "instance",
        "class_path": "test_autogalaxy.galaxy.test_dict.PixelizationGalaxy",
        "arguments": {
            "label": "cls345",
            "redshift": 1.0,
            "pixelization": {
                "type": "instance",
                "class_path": "autoarray.inversion.pixelization.pixelization.Pixelization",
                "arguments": {
                    "image_mesh": {
                        "type": "instance",
                        "class_path": "autoarray.inversion.pixelization.image_mesh.overlay.Overlay",
                        "arguments": {"shape": (3, 3)},
                    },
                    "mesh": {
                        "type": "instance",
                        "class_path": "autoarray.inversion.pixelization.mesh.voronoi.Voronoi",
                        "arguments": {},
                    },
                    "regularization": {
                        "type": "instance",
                        "class_path": "autoarray.inversion.regularization.adaptive_brightness.AdaptiveBrightness",
                        "arguments": {
                            "inner_coefficient": 1.0,
                            "outer_coefficient": 1.0,
                            "signal_scale": 1.0,
                        },
                    },
                },
            },
        },
    }


# TODO : Rich fix these to work with Pixelization objects (e.g. contained of mesh and regularization)


def test__with_pixelization__dict(pixelization_galaxy, pixelization_galaxy_dict):
    assert pixelization_galaxy.dict() == pixelization_galaxy_dict


def test_pixelization_equality():
    assert Voronoi() == Voronoi()


@pytest.fixture(name="profiles_galaxy")
def make_profiles_galaxy():
    return ag.Galaxy(redshift=2.0, light=ag.lp.Chameleon(), mass=ag.mp.DevVaucouleurs())


@pytest.fixture(name="profiles_galaxy_dict")
def make_profiles_galaxy_dict():
    return {
        "type": "instance",
        "class_path": "autogalaxy.galaxy.galaxy.Galaxy",
        "arguments": {
            "label": "cls345",
            "redshift": 2.0,
            "light": {
                "type": "instance",
                "class_path": "autogalaxy.profiles.light.standard.chameleon.Chameleon",
                "arguments": {
                    "centre": (0.0, 0.0),
                    "ell_comps": (0.0, 0.0),
                    "intensity": 0.1,
                    "core_radius_0": 0.01,
                    "core_radius_1": 0.05,
                },
            },
            "mass": {
                "type": "instance",
                "class_path": "autogalaxy.profiles.mass.stellar.dev_vaucouleurs.DevVaucouleurs",
                "arguments": {
                    "centre": (0.0, 0.0),
                    "ell_comps": (0.0, 0.0),
                    "intensity": 0.1,
                    "effective_radius": 0.6,
                    "mass_to_light_ratio": 1.0,
                },
            },
        },
    }


def test__profiles_galaxy__dict(profiles_galaxy, profiles_galaxy_dict):
    assert profiles_galaxy.dict() == profiles_galaxy_dict


def test__profiles_galaxy__from_dict(profiles_galaxy, profiles_galaxy_dict):
    assert from_dict(profiles_galaxy_dict) == profiles_galaxy
