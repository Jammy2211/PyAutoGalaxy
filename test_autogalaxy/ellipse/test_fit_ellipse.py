import numpy as np
import pytest

import autogalaxy as ag


def test__circular_radius():
    dataset = ag.DatasetEllipse(
        data=ag.Array2D.no_mask(
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
        ),
        noise_map=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
    )

    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=dataset, ellipse=ellipse_0)


def test__generic():
    dataset = ag.DatasetEllipse(
        data=ag.Array2D.no_mask(
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], pixel_scales=1.0
        ),
        noise_map=ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
    )

    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=dataset, ellipse=ellipse_0)

    print(fit.residual_map)
