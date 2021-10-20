import pytest

import autogalaxy as ag
from autogalaxy.dictable import Dictable


@pytest.fixture(
    name="trivial_galaxy"
)
def make_trivial_galaxy():
    return ag.Galaxy(
        redshift=1.0,
    )


@pytest.fixture(
    name="trivial_galaxy_dict"
)
def make_trivial_galaxy_dict():
    return {
        'hyper_galaxy': None,
        'pixelization': None,
        'redshift': 1.0,
        'regularization': None,
        'type': 'autogalaxy.galaxy.galaxy.Galaxy'
    }


def test_dict_trivial(
        trivial_galaxy,
        trivial_galaxy_dict
):
    assert trivial_galaxy.dict() == trivial_galaxy_dict


def test_from_dict(
        trivial_galaxy,
        trivial_galaxy_dict
):
    assert Dictable.from_dict(
        trivial_galaxy_dict
    ) == trivial_galaxy
