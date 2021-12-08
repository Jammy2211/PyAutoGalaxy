import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy import exc
from autogalaxy.mock import mock


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

    # TODO : Use VectorYX2D once merge complete.

    # data = ag.Array2D.manual_native(array=[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]], pixel_scales=1.0)
    data = np.array([[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]])

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


def test__apply_mask__masks_dataset(dataset_quantity_7x7_array_2d, sub_mask_2d_7x7):

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
