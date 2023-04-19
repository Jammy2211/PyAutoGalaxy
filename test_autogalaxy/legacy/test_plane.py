import numpy as np
import pytest

import autogalaxy as ag

from autoarray.inversion.pixelization.mesh.abstract import AbstractMesh


def test__adapt_galaxy_image_list():

    pixelization = ag.Pixelization(mesh=AbstractMesh())

    galaxy_pix = ag.legacy.Galaxy(redshift=0.5, pixelization=pixelization)

    plane = ag.legacy.Plane(galaxies=[galaxy_pix], redshift=None)
    assert plane.adapt_galaxies_with_pixelization_image_list[0] is None

    galaxy_pix = ag.legacy.Galaxy(
        redshift=0.5, pixelization=pixelization, adapt_galaxy_image=1
    )

    plane = ag.legacy.Plane(
        galaxies=[galaxy_pix, ag.legacy.Galaxy(redshift=0.5)], redshift=None
    )
    assert plane.adapt_galaxies_with_pixelization_image_list[0] == 1

    plane = ag.legacy.Plane(galaxies=[ag.legacy.Galaxy(redshift=0.5)], redshift=None)

    assert plane.adapt_galaxies_with_pixelization_image_list == []


def test__contribution_map_list():

    adapt_model_image = ag.Array2D.no_mask([[2.0, 4.0, 10.0]], pixel_scales=1.0)
    adapt_galaxy_image = ag.Array2D.no_mask([[1.0, 5.0, 8.0]], pixel_scales=1.0)

    hyper_galaxy_0 = ag.legacy.HyperGalaxy(contribution_factor=5.0)
    hyper_galaxy_1 = ag.legacy.HyperGalaxy(contribution_factor=10.0)

    contribution_map_0 = hyper_galaxy_0.contribution_map_from(
        adapt_model_image=adapt_model_image, adapt_galaxy_image=adapt_galaxy_image
    )

    contribution_map_1 = hyper_galaxy_1.contribution_map_from(
        adapt_model_image=adapt_model_image, adapt_galaxy_image=adapt_galaxy_image
    )

    galaxy_0 = ag.legacy.Galaxy(
        redshift=0.5,
        hyper_galaxy=hyper_galaxy_0,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image,
    )

    galaxy_1 = ag.legacy.Galaxy(
        redshift=0.5,
        hyper_galaxy=hyper_galaxy_1,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image,
    )

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[galaxy_0])

    assert (plane.contribution_map_list[0].slim == contribution_map_0).all()

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[galaxy_1])

    assert (plane.contribution_map_list[0].slim == contribution_map_1).all()

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[galaxy_1, galaxy_0])

    assert (plane.contribution_map_list[0].slim == contribution_map_1).all()
    assert (plane.contribution_map_list[1].slim == contribution_map_0).all()

    assert plane.contribution_map_list[0].slim == pytest.approx(
        [0.20833333, 0.89285714, 1.0], 1.0e-2
    )
    assert plane.contribution_map_list[1].slim == pytest.approx(
        [0.25714286, 1.0, 0.96], 1.0e-2
    )

    assert (sum(plane.contribution_map_list) == plane.contribution_map).all()

    adapt_model_image = ag.Array2D.no_mask([[2.0, 4.0, 10.0]], pixel_scales=1.0)
    adapt_galaxy_image = ag.Array2D.no_mask([[1.0, 5.0, 8.0]], pixel_scales=1.0)

    hyper_galaxy = ag.legacy.HyperGalaxy(contribution_factor=5.0)

    contribution_map = hyper_galaxy.contribution_map_from(
        adapt_model_image=adapt_model_image, adapt_galaxy_image=adapt_galaxy_image
    )

    galaxy = ag.legacy.Galaxy(
        redshift=0.5,
        hyper_galaxy=hyper_galaxy,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image,
    )

    plane = ag.legacy.Plane(
        redshift=0.5,
        galaxies=[
            galaxy,
            ag.legacy.Galaxy(redshift=0.5),
            ag.legacy.Galaxy(redshift=0.5),
        ],
    )

    assert (plane.contribution_map_list[0].slim == contribution_map).all()
    assert plane.contribution_map_list[1] == None
    assert plane.contribution_map_list[2] == None

    galaxy_1 = ag.legacy.Galaxy(redshift=0.5)

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[galaxy_0, galaxy_1])

    assert (galaxy_0.contribution_map == plane.contribution_map).all()

    galaxy_0 = ag.legacy.Galaxy(redshift=0.5)

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[galaxy_0, galaxy_1])

    assert plane.contribution_map == None


def test__hyper_noise_map_list_from():
    noise_map = ag.Array2D.no_mask(values=[[1.0, 2.0, 3.0]], pixel_scales=1.0)

    hyper_galaxy_0 = ag.legacy.HyperGalaxy(
        contribution_factor=0.0, noise_factor=1.0, noise_power=1.0
    )
    hyper_galaxy_1 = ag.legacy.HyperGalaxy(
        contribution_factor=3.0, noise_factor=1.0, noise_power=2.0
    )

    adapt_model_image = ag.Array2D.no_mask(values=[[0.5, 1.0, 1.5]], pixel_scales=1.0)

    adapt_galaxy_image_0 = ag.Array2D.no_mask(
        values=[[0.0, 1.0, 1.5]], pixel_scales=1.0
    )
    adapt_galaxy_image_1 = ag.Array2D.no_mask(
        values=[[1.0, 1.0, 1.5]], pixel_scales=1.0
    )

    galaxy_0 = ag.legacy.Galaxy(
        redshift=0.5,
        hyper_galaxy=hyper_galaxy_0,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image_0,
    )

    galaxy_1 = ag.legacy.Galaxy(
        redshift=0.5,
        hyper_galaxy=hyper_galaxy_1,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image_1,
    )

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[galaxy_0, galaxy_1])

    hyper_noise_map_list = plane.hyper_noise_map_list_from(noise_map=noise_map)

    assert (hyper_noise_map_list[0].slim == np.array([0.0, 2.0, 3.0])).all()
    assert hyper_noise_map_list[1].slim == pytest.approx(
        np.array([0.73468, (2.0 * 0.75) ** 2.0, 3.0**2.0]), 1.0e-4
    )

    noise_map = ag.Array2D.no_mask(values=[[5.0, 3.0, 1.0]], pixel_scales=1.0)

    adapt_model_image = ag.Array2D.no_mask(values=[[2.0, 4.0, 10.0]], pixel_scales=1.0)
    adapt_galaxy_image = ag.Array2D.no_mask(values=[[1.0, 5.0, 8.0]], pixel_scales=1.0)

    hyper_galaxy_0 = ag.legacy.HyperGalaxy(contribution_factor=5.0)
    hyper_galaxy_1 = ag.legacy.HyperGalaxy(contribution_factor=10.0)

    contribution_map_0 = hyper_galaxy_0.contribution_map_from(
        adapt_model_image=adapt_model_image, adapt_galaxy_image=adapt_galaxy_image
    )

    contribution_map_1 = hyper_galaxy_1.contribution_map_from(
        adapt_model_image=adapt_model_image, adapt_galaxy_image=adapt_galaxy_image
    )

    hyper_noise_map_0 = hyper_galaxy_0.hyper_noise_map_from(
        noise_map=noise_map, contribution_map=contribution_map_0
    )

    hyper_noise_map_1 = hyper_galaxy_1.hyper_noise_map_from(
        noise_map=noise_map, contribution_map=contribution_map_1
    )

    galaxy_0 = ag.legacy.Galaxy(
        redshift=0.5,
        hyper_galaxy=hyper_galaxy_0,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image,
    )

    galaxy_1 = ag.legacy.Galaxy(
        redshift=0.5,
        hyper_galaxy=hyper_galaxy_1,
        adapt_model_image=adapt_model_image,
        adapt_galaxy_image=adapt_galaxy_image,
    )

    plane = ag.legacy.Plane(
        redshift=0.5, galaxies=[galaxy_0, ag.legacy.Galaxy(redshift=0.5)]
    )

    hyper_noise_map_list = plane.hyper_noise_map_list_from(noise_map=noise_map)
    assert (hyper_noise_map_list[0].slim == hyper_noise_map_0).all()
    assert (hyper_noise_map_list[1].slim == np.zeros(shape=(3, 1))).all()

    plane = ag.legacy.Plane(
        redshift=0.5, galaxies=[ag.legacy.Galaxy(redshift=0.5), galaxy_1]
    )

    hyper_noise_map_list = plane.hyper_noise_map_list_from(noise_map=noise_map)
    assert (hyper_noise_map_list[0].slim == np.zeros(shape=(3, 1))).all()
    assert (hyper_noise_map_list[1].slim == hyper_noise_map_1).all()

    plane = ag.legacy.Plane(
        redshift=0.5,
        galaxies=[
            ag.legacy.Galaxy(redshift=0.5),
            galaxy_1,
            galaxy_0,
            ag.legacy.Galaxy(redshift=0.5),
        ],
    )

    hyper_noise_map_list = plane.hyper_noise_map_list_from(noise_map=noise_map)
    assert (hyper_noise_map_list[0].slim == np.zeros(shape=(3, 1))).all()
    assert (hyper_noise_map_list[1].slim == hyper_noise_map_1).all()
    assert (hyper_noise_map_list[2].slim == hyper_noise_map_0).all()
    assert (hyper_noise_map_list[3].slim == np.zeros(shape=(3, 1))).all()

    # Filters Nones from list when computing noise map.

    plane = ag.legacy.Plane(
        redshift=0.5,
        galaxies=[
            ag.legacy.Galaxy(redshift=0.5),
            galaxy_1,
            galaxy_0,
            ag.legacy.Galaxy(redshift=0.5),
        ],
    )

    hyper_noise_map = plane.hyper_noise_map_from(noise_map=noise_map)
    assert (hyper_noise_map.slim == hyper_noise_map_0 + hyper_noise_map_1).all()

    # No Galaxies

    noise_map = ag.Array2D.no_mask(values=[[5.0, 3.0, 1.0]], pixel_scales=1.0)

    plane = ag.legacy.Plane(redshift=0.5, galaxies=[ag.legacy.Galaxy(redshift=0.5)])
    hyper_noise_map = plane.hyper_noise_map_from(noise_map=noise_map)

    assert (hyper_noise_map == np.zeros((3, 1))).all()
