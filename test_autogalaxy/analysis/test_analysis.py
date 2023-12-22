import pytest
import numpy as np
import os
from os import path

from autoconf.dictable import from_json

import autofit as af
import autogalaxy as ag

directory = path.dirname(path.realpath(__file__))


def test__plane_via_instance(masked_imaging_7x7):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))
    clump = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.2))

    model = af.Collection(
        galaxies=af.Collection(galaxy=galaxy), clumps=af.Collection(clump_0=clump)
    )

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    instance = model.instance_from_unit_vector([])

    plane = analysis.plane_via_instance_from(instance=instance)

    assert plane.galaxies[0].light.intensity == 0.1
    assert plane.galaxies[1].light.intensity == 0.2


def test__instance_with_associated_adapt_images_from(masked_imaging_7x7):
    galaxies = af.ModelInstance()
    galaxies.galaxy = ag.Galaxy(redshift=0.5)
    galaxies.source = ag.Galaxy(redshift=1.0)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    adapt_galaxy_name_image_dict = {
        str(("galaxies", "galaxy")): ag.Array2D.ones(
            shape_native=(3, 3), pixel_scales=1.0
        ),
        str(("galaxies", "source")): ag.Array2D.full(
            fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    }

    adapt_images = ag.AdaptImages(
        model_image=ag.Array2D.full(
            fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
        ),
        galaxy_name_image_dict=adapt_galaxy_name_image_dict,
    )

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, adapt_images=adapt_images)

    adapt_images = analysis.adapt_images_via_instance_from(instance=instance)

    assert adapt_images.model_image.native == pytest.approx(
        3.0 * np.ones((3, 3)), 1.0e-4
    )

    assert adapt_images.galaxy_image_dict[galaxies.galaxy].native == pytest.approx(
        np.ones((3, 3)), 1.0e-4
    )
    assert adapt_images.galaxy_image_dict[galaxies.source].native == pytest.approx(
        2.0 * np.ones((3, 3)), 1.0e-4
    )


def test__modify_before_fit__kmeans_pixelization_upper_limit_ajusted_based_on_mask(
    masked_imaging_7x7,
):
    image_mesh = af.Model(ag.image_mesh.KMeans)
    image_mesh.pixels = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)

    pixelization = af.Model(ag.Pixelization, mesh=None, image_mesh=image_mesh)

    galaxies = af.Collection(source=ag.Galaxy(redshift=0.5, pixelization=pixelization))

    model = af.Collection(galaxies=galaxies)

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    analysis.modify_before_fit(paths=af.DirectoryPaths(), model=model)

    assert (
        model.galaxies.source.pixelization.image_mesh.pixels.upper_limit
        == pytest.approx(9, 1.0e-4)
    )


def test__save_results__plane_output_to_json(analysis_imaging_7x7):
    galaxy = ag.Galaxy(redshift=0.5)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    plane = ag.Plane(galaxies=[galaxy])

    paths = af.DirectoryPaths()

    analysis_imaging_7x7.save_results(
        paths=paths, result=ag.m.MockResult(max_log_likelihood_plane=plane, model=model)
    )

    plane = from_json(file_path=paths._files_path / "plane.json")

    assert plane.galaxies[0].redshift == 0.5

    os.remove(paths._files_path / "plane.json")
