import numpy as np

import autogalaxy as ag


def test__contribution_map_from():

    hyper_image = np.ones((3,))

    hyp = ag.legacy.HyperGalaxy(contribution_factor=0.0)
    contribution_map = hyp.contribution_map_from(
        adapt_model_image=hyper_image, adapt_galaxy_image=hyper_image
    )

    assert (contribution_map == np.ones((3,))).all()

    hyper_image = np.ones((3,))

    hyp = ag.legacy.HyperGalaxy(contribution_factor=0.0)

    galaxy = ag.legacy.Galaxy(
        redshift=0.5,
        hyper_galaxy=hyp,
        adapt_galaxy_image=hyper_image,
        adapt_model_image=hyper_image,
    )

    contribution_map = hyp.contribution_map_from(
        adapt_model_image=hyper_image, adapt_galaxy_image=hyper_image
    )

    assert (contribution_map == galaxy.contribution_map).all()


def test__hyper_noise_map_from():

    noise_map = np.array([1.0, 2.0, 3.0])
    contribution_map = np.array([[0.0, 0.5, 1.0]])

    hyper_galaxy = ag.legacy.HyperGalaxy(
        contribution_factor=0.0, noise_factor=2.0, noise_power=2.0
    )

    hyper_noise_map = hyper_galaxy.hyper_noise_map_from(
        noise_map=noise_map, contribution_map=contribution_map
    )

    assert (hyper_noise_map == np.array([0.0, 2.0, 18.0])).all()
