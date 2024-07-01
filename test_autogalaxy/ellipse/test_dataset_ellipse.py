import numpy as np
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

def test__data_interp():

    dataset = ag.DatasetEllipse(
        data=ag.Array2D.no_mask(values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0),
        noise_map=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        radii_min=1.0,
        radii_max=2.0,
        radii_bins=3,
    )

    assert dataset.data_interp((1.5, 1.5)) == pytest.approx(7.0, 1.0e-4)