import pytest
import numpy as np
from os import path

import autofit as af
import autogalaxy as ag

directory = path.dirname(path.realpath(__file__))


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
        galaxy_name_image_dict=adapt_galaxy_name_image_dict,
    )

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)
    analysis._adapt_images = adapt_images

    adapt_images = analysis.adapt_images_via_instance_from(instance=instance)

    assert adapt_images.galaxy_image_dict[galaxies.galaxy].native == pytest.approx(
        np.ones((3, 3)), 1.0e-4
    )
    assert adapt_images.galaxy_image_dict[galaxies.source].native == pytest.approx(
        2.0 * np.ones((3, 3)), 1.0e-4
    )
