import pytest

import autogalaxy as ag

def test__radii_array():

    dataset = ag.DatasetEllipse(
        data=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        noise_map=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        radii_min=1.0,
        radii_max=2.0,
        radii_bins=3,
    )

    assert dataset.radii_array == pytest.approx([10.0, 31.6227766, 100.0], 1.0e-4)