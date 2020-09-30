from os import path
from os.path import dirname, realpath

import numpy as np
import pytest
from matplotlib import pyplot

import autoarray as aa
from autoconf import conf


class PlotPatch:
    def __init__(self):
        self.paths = []

    def __call__(self, path, *args, **kwargs):
        self.paths.append(path)


@pytest.fixture(name="plot_patch")
def make_plot_patch(monkeypatch):
    plot_patch = PlotPatch()
    monkeypatch.setattr(pyplot, "savefig", plot_patch)
    return plot_patch


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(autouse=True)
def set_config_path(request):
    if dirname(realpath(__file__)) in str(request.module):
        conf.instance = conf.Config.for_directory(
            directory
        )


@pytest.fixture(name="visibilities_7x2")
def make_visibilities_7():
    visibilities = aa.Visibilities.full(shape_1d=(7,), fill_value=1.0)
    visibilities[6, 0] = -1.0
    visibilities[6, 1] = -1.0
    return visibilities


@pytest.fixture(name="noise_map_7x2")
def make_noise_map_7():
    return aa.VisibilitiesNoiseMap.full(shape_1d=(7,), fill_value=2.0)


@pytest.fixture(name="uv_wavelengths_7x2")
def make_uv_wavelengths_7():
    return np.array(
        [
            [-55636.4609375, 171376.90625],
            [-6903.21923828, 51155.578125],
            [-63488.4140625, 4141.28369141],
            [55502.828125, 47016.7265625],
            [54160.75390625, -99354.1796875],
            [-9327.66308594, -95212.90625],
            [0.0, 0.0],
        ]
    )


@pytest.fixture(name="blurring_grid_7x7")
def make_blurring_grid_7x7(blurring_mask_7x7):
    return aa.Grid.from_mask(mask=blurring_mask_7x7)


@pytest.fixture(name="blurring_mask_7x7")
def make_blurring_mask_7x7():
    blurring_mask = np.array(
        [
            [True, True, True, True, True, True, True],
            [True, False, False, False, False, False, True],
            [True, False, True, True, True, False, True],
            [True, False, True, True, True, False, True],
            [True, False, True, True, True, False, True],
            [True, False, False, False, False, False, True],
            [True, True, True, True, True, True, True],
        ]
    )

    return aa.Mask2D.manual(mask=blurring_mask, pixel_scales=(1.0, 1.0))
