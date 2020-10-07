from os import path
from os.path import dirname, realpath

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
    return aa.mock.make_visibilities_7()


@pytest.fixture(name="noise_map_7x2")
def make_noise_map_7():
    return aa.mock.make_noise_map_7()


@pytest.fixture(name="uv_wavelengths_7x2")
def make_uv_wavelengths_7():
    return aa.mock.make_uv_wavelengths_7()


@pytest.fixture(name="blurring_grid_7x7")
def make_blurring_grid_7x7(blurring_mask_7x7):
    return aa.Grid.from_mask(mask=blurring_mask_7x7)


@pytest.fixture(name="blurring_mask_7x7")
def make_blurring_mask_7x7():
    return aa.mock.make_blurring_mask_7x7()
