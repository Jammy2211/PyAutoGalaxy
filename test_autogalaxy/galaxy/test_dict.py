import autogalaxy as ag


def test_dict_trivial():
    galaxy = ag.Galaxy(
        redshift=1.0,
    )
    assert galaxy.dict() == {
        'hyper_galaxy': None,
        'pixelization': None,
        'redshift': 1.0,
        'regularization': None,
        'type': 'autogalaxy.galaxy.galaxy.Galaxy'
    }
