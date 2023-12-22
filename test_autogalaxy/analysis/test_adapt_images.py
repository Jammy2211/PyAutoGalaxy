import pytest
import numpy as np

import autogalaxy as ag


def test__instance_with_associated_adapt_images_from(masked_imaging_7x7):
    g0 = ag.Galaxy(redshift=0.5)
    g1 = ag.Galaxy(redshift=1.0)

    galaxy_image_dict = {
        g0: ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        g1: ag.Array2D.full(fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0),
    }

    adapt_images = ag.AdaptImages(
        galaxy_image_dict=galaxy_image_dict,
    )

    assert adapt_images.model_image.native == pytest.approx(
        3.0 * np.ones((3, 3)), 1.0e-4
    )
