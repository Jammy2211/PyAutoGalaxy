import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.mock.mock import MockMassProfile


def test__fit_via_mock_profile(dataset_quantity_7x7_array_2d):

    model_object = MockMassProfile(
        convergence_2d=ag.Array2D.ones((7, 7), pixel_scales=1.0)
    )

    fit_quantity = ag.FitQuantity(
        dataset_quantity=dataset_quantity_7x7_array_2d,
        model_func=model_object.convergence_2d_from,
    )

    assert fit_quantity.chi_squared == pytest.approx(0.0, 1.0e-4)

    assert fit_quantity.log_likelihood == pytest.approx(
        -0.5 * 49.0 * np.log(2 * np.pi * 2.0 ** 2.0), 1.0e-4
    )
