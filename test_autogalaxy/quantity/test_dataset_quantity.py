import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy import exc


def test_via_signal_to_noise_map(dataset_quantity_7x7_array_2d, sub_mask_2d_7x7):

    data = ag.Array2D.manual_native(array=[[1.0, 2.0], [3.0, 4.0]], pixel_scales=1.0)
    signal_to_noise_map = ag.Array2D.manual_native(
        array=[[1.0, 5.0], [15.0, 40.0]], pixel_scales=1.0
    )

    dataset_quantity = ag.DatasetQuantity.via_signal_to_noise_map(
        data=data, signal_to_noise_map=signal_to_noise_map
    )

    assert dataset_quantity.signal_to_noise_map == pytest.approx(
        signal_to_noise_map, 1.0e-4
    )
    assert dataset_quantity.noise_map.native == pytest.approx(
        np.array([[1.0, 0.4], [0.2, 0.1]]), 1.0e-4
    )

    data = ag.VectorYX2D.manual_native(
        vectors=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], pixel_scales=1.0
    )
    signal_to_noise_map = ag.Array2D.manual_native(
        array=[[1.0, 5.0], [15.0, 40.0]], pixel_scales=1.0
    )

    dataset_quantity = ag.DatasetQuantity.via_signal_to_noise_map(
        data=data, signal_to_noise_map=signal_to_noise_map
    )

    assert dataset_quantity.signal_to_noise_map == pytest.approx(
        np.array([[1.0, 1.0], [5.0, 5.0], [15.0, 15.0], [40.0, 40.0]]), 1.0e-4
    )
    assert dataset_quantity.noise_map.native == pytest.approx(
        np.array([[[1.0, 1.0], [0.4, 0.4]], [[0.2, 0.2], [0.1, 0.1]]]), 1.0e-4
    )


def test__apply_mask__masks_dataset(
    dataset_quantity_7x7_array_2d, dataset_quantity_7x7_vector_yx_2d, sub_mask_2d_7x7
):

    dataset_quantity_7x7 = dataset_quantity_7x7_array_2d.apply_mask(
        mask=sub_mask_2d_7x7
    )

    assert (dataset_quantity_7x7.data.slim == np.ones(9)).all()
    assert (
        dataset_quantity_7x7.data.native == np.ones((7, 7)) * np.invert(sub_mask_2d_7x7)
    ).all()

    assert (dataset_quantity_7x7.noise_map.slim == 2.0 * np.ones(9)).all()
    assert (
        dataset_quantity_7x7.noise_map.native
        == 2.0 * np.ones((7, 7)) * np.invert(sub_mask_2d_7x7)
    ).all()

    dataset_quantity_7x7 = dataset_quantity_7x7_vector_yx_2d.apply_mask(
        mask=sub_mask_2d_7x7
    )

    assert (dataset_quantity_7x7.data.slim == np.ones((9, 2))).all()
    assert (dataset_quantity_7x7.noise_map.slim == 2.0 * np.ones((9, 2))).all()


def test__grid(
    dataset_quantity_7x7_array_2d,
    sub_mask_2d_7x7,
    grid_2d_7x7,
    sub_grid_2d_7x7,
    blurring_grid_2d_7x7,
    grid_2d_iterate_7x7,
):
    masked_imaging_7x7 = dataset_quantity_7x7_array_2d.apply_mask(mask=sub_mask_2d_7x7)
    masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
        settings=ag.SettingsImaging(grid_class=ag.Grid2D, sub_size=2)
    )

    assert isinstance(masked_imaging_7x7.grid, ag.Grid2D)
    assert (masked_imaging_7x7.grid.binned == grid_2d_7x7).all()
    assert (masked_imaging_7x7.grid.slim == sub_grid_2d_7x7).all()

    masked_imaging_7x7 = dataset_quantity_7x7_array_2d.apply_mask(mask=sub_mask_2d_7x7)
    masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
        settings=ag.SettingsImaging(grid_class=ag.Grid2DIterate)
    )

    assert isinstance(masked_imaging_7x7.grid, ag.Grid2DIterate)
    assert (masked_imaging_7x7.grid.binned == grid_2d_iterate_7x7).all()


def test__vector_data__y_x():

    data = ag.VectorYX2D.manual_native(
        vectors=[[[1.0, 5.0], [2.0, 6.0]], [[3.0, 7.0], [4.0, 8.0]]],
        pixel_scales=1.0,
        sub_size=1,
    )

    noise_map = ag.VectorYX2D.manual_native(
        vectors=[[[1.1, 5.1], [2.1, 6.1]], [[3.1, 7.1], [4.1, 8.1]]],
        pixel_scales=1.0,
        sub_size=1,
    )

    dataset_quantity = ag.DatasetQuantity(data=data, noise_map=noise_map)

    assert isinstance(dataset_quantity.y, ag.DatasetQuantity)
    assert (dataset_quantity.y.data.slim == np.array([1.0, 2.0, 3.0, 4.0])).all()
    assert (dataset_quantity.y.noise_map.slim == np.array([1.1, 2.1, 3.1, 4.1])).all()

    assert isinstance(dataset_quantity.y, ag.DatasetQuantity)
    assert (dataset_quantity.x.data.slim == np.array([5.0, 6.0, 7.0, 8.0])).all()
    assert (dataset_quantity.x.noise_map.slim == np.array([5.1, 6.1, 7.1, 8.1])).all()
