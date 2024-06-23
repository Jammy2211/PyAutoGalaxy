import numpy as np
import pytest

import autogalaxy as ag


def test__adapt_over_sample__used_if_no_over_sampling_input(gal_x1_lp):
    # In grid.yaml this class has settings which use the autoarray over sampling adaptive decorator.

    class SersicAdaptTest(ag.lp.Sersic):
        pass

    mask = ag.Mask2D.circular(
        radius=1.0,
        shape_native=(21, 21),
        pixel_scales=0.1,
    )

    light = SersicAdaptTest(intensity=1.0)

    grid = ag.Grid2D.from_mask(mask=mask, over_sampling=None)

    image = light.image_2d_from(grid=grid)

    assert np.max(image) == pytest.approx(53.83706341021047, 1.0e-6)
