import numpy as np
import pytest

import autogalaxy as ag


def test__fit_via_mock_profile__convergence__chi_squared_zero_and_correct_log_likelihood(
    dataset_quantity_7x7_array_2d,
):
    mass = ag.m.MockMassProfile(
        convergence_2d=ag.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0),
        potential_2d=ag.Array2D.full(
            fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0
        ),
        deflections_yx_2d=ag.VectorYX2D.full(
            fill_value=3.0, shape_native=(7, 7), pixel_scales=1.0
        ),
    )

    galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5, mass=mass)])

    fit_quantity = ag.FitQuantity(
        dataset=dataset_quantity_7x7_array_2d,
        light_mass_obj=galaxies,
        func_str="convergence_2d_from",
    )

    assert fit_quantity.chi_squared == pytest.approx(0.0, 1.0e-4)
    assert fit_quantity.log_likelihood == pytest.approx(
        -0.5 * 49.0 * np.log(2 * np.pi * 2.0**2.0), 1.0e-4
    )


def test__fit_via_mock_profile__potential__nonzero_chi_squared_and_correct_log_likelihood(
    dataset_quantity_7x7_array_2d,
):
    mass = ag.m.MockMassProfile(
        convergence_2d=ag.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0),
        potential_2d=ag.Array2D.full(
            fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0
        ),
        deflections_yx_2d=ag.VectorYX2D.full(
            fill_value=3.0, shape_native=(7, 7), pixel_scales=1.0
        ),
    )

    galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5, mass=mass)])

    fit_quantity = ag.FitQuantity(
        dataset=dataset_quantity_7x7_array_2d,
        light_mass_obj=galaxies,
        func_str="potential_2d_from",
    )

    assert fit_quantity.chi_squared == pytest.approx(12.25, 1.0e-4)
    assert fit_quantity.log_likelihood == pytest.approx(-85.1171999, 1.0e-4)


def test__fit_via_mock_profile__deflections__vector_dataset__correct_chi_squared_and_log_likelihood(
    dataset_quantity_7x7_vector_yx_2d,
):
    mass = ag.m.MockMassProfile(
        convergence_2d=ag.Array2D.ones(shape_native=(7, 7), pixel_scales=1.0),
        potential_2d=ag.Array2D.full(
            fill_value=2.0, shape_native=(7, 7), pixel_scales=1.0
        ),
        deflections_yx_2d=ag.VectorYX2D.full(
            fill_value=3.0, shape_native=(7, 7), pixel_scales=1.0
        ),
    )

    galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5, mass=mass)])

    fit_quantity = ag.FitQuantity(
        dataset=dataset_quantity_7x7_vector_yx_2d,
        light_mass_obj=galaxies,
        func_str="deflections_yx_2d_from",
    )

    assert fit_quantity.chi_squared == pytest.approx(98.0, 1.0e-4)
    assert fit_quantity.log_likelihood == pytest.approx(-206.98438, 1.0e-4)


def test__y_x(dataset_quantity_7x7_vector_yx_2d):
    model_object = ag.m.MockMassProfile(
        deflections_yx_2d=ag.VectorYX2D.full(
            fill_value=3.0, shape_native=(7, 7), pixel_scales=1.0
        )
    )

    galaxies = ag.Galaxies(galaxies=[ag.Galaxy(redshift=0.5, mass=model_object)])

    fit_quantity = ag.FitQuantity(
        dataset=dataset_quantity_7x7_vector_yx_2d,
        light_mass_obj=galaxies,
        func_str="deflections_yx_2d_from",
    )

    assert (
        fit_quantity.y.dataset.data == dataset_quantity_7x7_vector_yx_2d.y.data
    ).all()
    assert (
        fit_quantity.x.dataset.data == dataset_quantity_7x7_vector_yx_2d.x.data
    ).all()
