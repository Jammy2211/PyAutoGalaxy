from matplotlib.patches import Ellipse
import pytest
import numpy as np

import autogalaxy as ag


def test__elliptical_properties_and_patches():

    vector_field = ag.ShearField2DIrregular(
        vectors=[(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)],
        grid=[[1.0, -1.0], [1.0, 1.0], [0.0, 0.0]],
    )

    assert isinstance(vector_field.ellipticities, ag.ValuesIrregular)
    assert vector_field.ellipticities.in_list == [1.0, 1.0, np.sqrt(2.0)]

    assert isinstance(vector_field.semi_major_axes, ag.ValuesIrregular)
    assert vector_field.semi_major_axes.in_list == pytest.approx(
        [6.0, 6.0, 7.242640], 1.0e-4
    )

    assert isinstance(vector_field.semi_minor_axes, ag.ValuesIrregular)
    assert vector_field.semi_minor_axes.in_list == pytest.approx(
        [0.0, 0.0, -1.242640], 1.0e-4
    )

    assert isinstance(vector_field.phis, ag.ValuesIrregular)
    assert vector_field.phis.in_list == pytest.approx([0.0, 45.0, 22.5], 1.0e-4)

    assert isinstance(vector_field.elliptical_patches[0], Ellipse)
    assert vector_field.elliptical_patches[1].center == pytest.approx(
        (1.0, 1.0), 1.0e-4
    )
    assert vector_field.elliptical_patches[1].width == pytest.approx(6.0, 1.0e-4)
    assert vector_field.elliptical_patches[1].height == pytest.approx(0.0, 1.0e-4)
    assert vector_field.elliptical_patches[1].angle == pytest.approx(45.0, 1.0e-4)
