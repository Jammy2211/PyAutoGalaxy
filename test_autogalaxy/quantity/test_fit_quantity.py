import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.mock.mock import MockMassProfile


def test__fit_via_mock_profile(dataset_quantity_7x7_array_2d):

    model_object = MockMassProfile(
        convergence_2d=ag.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0),
        potential_2d=ag.Array2D.full(
            fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0
        ),
    )

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5, mass=model_object)])

    fit_quantity = ag.FitQuantity(
        dataset=dataset_quantity_7x7_array_2d,
        light_mass_obj=plane,
        func_str="convergence_2d_from",
    )

    assert fit_quantity.chi_squared == pytest.approx(0.0, 1.0e-4)

    assert fit_quantity.log_likelihood == pytest.approx(
        -0.5 * 49.0 * np.log(2 * np.pi * 2.0 ** 2.0), 1.0e-4
    )

    fit_quantity = ag.FitQuantity(
        dataset=dataset_quantity_7x7_array_2d,
        light_mass_obj=plane,
        func_str="potential_2d_from",
    )

    assert fit_quantity.chi_squared == pytest.approx(12.25, 1.0e-4)

    assert fit_quantity.log_likelihood == pytest.approx(-85.1171999, 1.0e-4)