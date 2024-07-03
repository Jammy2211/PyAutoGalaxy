import numpy as np
import pytest

import autogalaxy as ag


def test__data_interp():
    dataset = ag.DatasetEllipse(
        data=ag.Array2D.no_mask(
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
        ),
        noise_map=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
    )

    assert dataset.data_interp((1.5, 1.5)) == pytest.approx(7.0, 1.0e-4)


def test__noise_map_interp():
    dataset = ag.DatasetEllipse(
        data=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
        noise_map=ag.Array2D.no_mask(
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
        ),
    )

    assert dataset.noise_map_interp((1.5, 1.5)) == pytest.approx(7.0, 1.0e-4)
