import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy import exc
from autogalaxy.mock import mock


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
