import pytest
import numpy as np
from os import path

import autofit as af
import autogalaxy as ag

directory = path.dirname(path.realpath(__file__))


def test__plane_via_instance(masked_imaging_7x7):

    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=0.1))
    clump = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=0.2))

    model = af.Collection(
        galaxies=af.Collection(galaxy=galaxy), clumps=af.Collection(clump_0=clump)
    )

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    instance = model.instance_from_unit_vector([])

    plane = analysis.plane_via_instance_from(instance=instance)

    assert plane.galaxies[0].light.intensity == 0.1
    assert plane.galaxies[1].light.intensity == 0.2


def test__instance_with_associated_hyper_images_from(masked_imaging_7x7):

    galaxies = af.ModelInstance()
    galaxies.galaxy = ag.Galaxy(redshift=0.5)
    galaxies.source = ag.Galaxy(redshift=1.0)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    hyper_galaxy_image_path_dict = {
        ("galaxies", "galaxy"): ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        ("galaxies", "source"): ag.Array2D.full(
            fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    }

    result = ag.m.MockResult(
        instance=instance,
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
        hyper_model_image=ag.Array2D.full(
            fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    )

    analysis = ag.AnalysisImaging(
        dataset=masked_imaging_7x7, hyper_dataset_result=result
    )

    instance = analysis.instance_with_associated_hyper_images_from(instance=instance)

    assert instance.galaxies.galaxy.hyper_galaxy_image.native == pytest.approx(
        np.ones((3, 3)), 1.0e-4
    )
    assert instance.galaxies.source.hyper_galaxy_image.native == pytest.approx(
        2.0 * np.ones((3, 3)), 1.0e-4
    )

    assert instance.galaxies.galaxy.hyper_model_image.native == pytest.approx(
        3.0 * np.ones((3, 3)), 1.0e-4
    )
    assert instance.galaxies.source.hyper_model_image.native == pytest.approx(
        3.0 * np.ones((3, 3)), 1.0e-4
    )
