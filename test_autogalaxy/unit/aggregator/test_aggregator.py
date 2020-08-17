import os
from os import path

import autofit as af
import autogalaxy as ag
import numpy as np
import pytest
from test_autogalaxy import mock

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="path")
def make_path():
    return "{}/files/".format(os.path.dirname(os.path.realpath(__file__)))


@pytest.fixture(name="samples")
def make_samples():

    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(centre=(0.0, 1.0)))
    galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.EllipticalSersic())

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    return mock.MockSamples(max_log_likelihood_instance=plane)


def test__plane_generator_from_aggregator(imaging_7x7, mask_7x7, samples):

    phase_imaging_7x7 = ag.PhaseImaging(
        phase_name="test_phase_aggregator",
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        search=mock.MockSearch(samples=samples),
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults(samples=samples)
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    plane_gen = ag.agg.Plane(aggregator=agg)

    for plane in plane_gen:

        assert plane.galaxies[0].redshift == 0.5
        assert plane.galaxies[0].light.centre == (0.0, 1.0)
        assert plane.galaxies[1].redshift == 1.0


def test__masked_imaging_generator_from_aggregator(imaging_7x7, mask_7x7, samples):

    phase_imaging_7x7 = ag.PhaseImaging(
        phase_name="test_phase_aggregator",
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        settings=ag.PhaseSettingsImaging(
            masked_imaging_settings=ag.MaskedImagingSettings(
                grid_class=ag.GridIterate,
                grid_inversion_class=ag.GridIterate,
                fractional_accuracy=0.5,
                sub_steps=[2],
            )
        ),
        search=mock.MockSearch(samples=samples),
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults(samples=samples)
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    masked_imaging_gen = ag.agg.MaskedImaging(aggregator=agg)

    for masked_imaging in masked_imaging_gen:
        assert (masked_imaging.imaging.image == imaging_7x7.image).all()
        assert isinstance(masked_imaging.grid, ag.GridIterate)
        assert isinstance(masked_imaging.grid_inversion, ag.GridIterate)
        assert masked_imaging.grid.sub_steps == [2]
        assert masked_imaging.grid.fractional_accuracy == 0.5


def test__fit_imaging_generator_from_aggregator(imaging_7x7, mask_7x7, samples):

    phase_imaging_7x7 = ag.PhaseImaging(
        phase_name="test_phase_aggregator",
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        search=mock.MockSearch(samples=samples),
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults(samples=samples)
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    fit_imaging_gen = ag.agg.FitImaging(aggregator=agg)

    for fit_imaging in fit_imaging_gen:
        assert (fit_imaging.masked_imaging.imaging.image == imaging_7x7.image).all()


def test__masked_interferometer_generator_from_aggregator(
    interferometer_7, mask_7x7, samples
):

    phase_interferometer_7x7 = ag.PhaseInterferometer(
        phase_name="test_phase_aggregator",
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        settings=ag.PhaseSettingsInterferometer(
            masked_interferometer_settings=ag.MaskedInterferometerSettings(
                grid_class=ag.GridIterate,
                grid_inversion_class=ag.GridIterate,
                fractional_accuracy=0.5,
                sub_steps=[2],
                transformer_class=ag.TransformerDFT,
            )
        ),
        search=mock.MockSearch(samples=samples),
        real_space_mask=mask_7x7,
    )

    phase_interferometer_7x7.run(
        dataset=interferometer_7,
        mask=mask_7x7,
        results=mock.MockResults(samples=samples),
    )

    agg = af.Aggregator(directory=phase_interferometer_7x7.paths.output_path)

    masked_interferometer_gen = ag.agg.MaskedInterferometer(aggregator=agg)

    for masked_interferometer in masked_interferometer_gen:
        assert (
            masked_interferometer.interferometer.visibilities
            == interferometer_7.visibilities
        ).all()
        assert (masked_interferometer.real_space_mask == mask_7x7).all()
        assert isinstance(masked_interferometer.grid, ag.GridIterate)
        assert isinstance(masked_interferometer.grid_inversion, ag.GridIterate)
        assert masked_interferometer.grid.sub_steps == [2]
        assert masked_interferometer.grid.fractional_accuracy == 0.5
        assert isinstance(masked_interferometer.transformer, ag.TransformerDFT)


def test__fit_interferometer_generator_from_aggregator(
    interferometer_7, mask_7x7, samples
):

    phase_interferometer_7x7 = ag.PhaseInterferometer(
        phase_name="test_phase_aggregator",
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        search=mock.MockSearch(samples=samples),
        real_space_mask=mask_7x7,
    )

    phase_interferometer_7x7.run(
        dataset=interferometer_7,
        mask=mask_7x7,
        results=mock.MockResults(samples=samples),
    )

    agg = af.Aggregator(directory=phase_interferometer_7x7.paths.output_path)

    fit_interferometer_gen = ag.agg.FitInterferometer(aggregator=agg)

    for fit_interferometer in fit_interferometer_gen:
        assert (
            fit_interferometer.masked_interferometer.interferometer.visibilities
            == interferometer_7.visibilities
        ).all()
        assert (
            fit_interferometer.masked_interferometer.real_space_mask == mask_7x7
        ).all()


class MockResult:
    def __init__(self, log_likelihood):
        self.log_likelihood = log_likelihood
        self.model = log_likelihood


class MockAggregator:
    def __init__(self, grid_search_result):

        self.grid_search_result = grid_search_result

    @property
    def grid_search_results(self):
        return iter([self.grid_search_result])

    def values(self, str):
        return self.grid_search_results


def test__results_array_from_results_file(path):

    results = [
        MockResult(log_likelihood=1.0),
        MockResult(log_likelihood=2.0),
        MockResult(log_likelihood=3.0),
        MockResult(log_likelihood=4.0),
    ]

    lower_limit_lists = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]]
    physical_lower_limits_lists = [[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]

    grid_search_result = af.GridSearchResult(
        results=results,
        physical_lower_limits_lists=physical_lower_limits_lists,
        lower_limit_lists=lower_limit_lists,
    )

    aggregator = MockAggregator(grid_search_result=grid_search_result)

    array = ag.agg.grid_search_result_as_array(aggregator=aggregator)

    assert array.in_2d == pytest.approx(np.array([[3.0, 2.0], [1.0, 4.0]]), 1.0e4)
    assert array.pixel_scales == (1.0, 1.0)


# def test__results_array_from_real_grid_search_pickle(path):
#
#     with open("{}/{}.pickle".format(path, "grid_search_result"), "rb") as f:
#         grid_search_result = pickle.load(f)
#
#     assert grid_search_result.physical_step_sizes == pytest.approx((0.8, 0.8), 1.0e-4)
#
#     array = ag.agg.grid_search_result_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     assert array.in_2d[0, 0] == pytest.approx(21039.54, 1.0e-2)
#     assert array.in_2d[0, 1] == pytest.approx(21040.64, 1.0e-2)
#     assert array.in_2d[0, 2] == pytest.approx(21629.03, 1.0e-2)
#     assert array.in_2d[0, 3] == pytest.approx(21623.81, 1.0e-2)
#     assert array.in_2d[0, 4] == pytest.approx(21039.58, 1.0e-2)
#     assert array.in_2d[1, 0] == pytest.approx(21052.43, 1.0e-2)
#     assert array.in_2d[1, 1] == pytest.approx(21039.60, 1.0e-2)
#     assert array.in_2d[1, 2] == pytest.approx(21086.19, 1.0e-2)
#     assert array.in_2d[1, 3] == pytest.approx(21070.26, 1.0e-2)
#     assert array.in_2d[1, 4] == pytest.approx(21039.44, 1.0e-2)
#     assert array.in_2d[2, 0] == pytest.approx(21068.41, 1.0e-2)
#     assert array.in_2d[2, 1] == pytest.approx(21046.94, 1.0e-2)
#     assert array.in_2d[2, 2] == pytest.approx(21039.71, 1.0e-2)
#     assert array.in_2d[2, 3] == pytest.approx(21048.62, 1.0e-2)
#     assert array.in_2d[2, 4] == pytest.approx(21064.88, 1.0e-2)
#     assert array.in_2d[3, 0] == pytest.approx(21064.22, 1.0e-2)
#     assert array.in_2d[3, 1] == pytest.approx(21046.82, 1.0e-2)
#     assert array.in_2d[3, 2] == pytest.approx(21039.69, 1.0e-2)
#     assert array.in_2d[3, 3] == pytest.approx(21051.30, 1.0e-2)
#     assert array.in_2d[3, 4] == pytest.approx(21064.96, 1.0e-2)
#     assert array.in_2d[4, 0] == pytest.approx(21047.51, 1.0e-2)
#     assert array.in_2d[4, 1] == pytest.approx(21039.91, 1.0e-2)
#     assert array.in_2d[4, 2] == pytest.approx(21039.49, 1.0e-2)
#     assert array.in_2d[4, 3] == pytest.approx(21048.40, 1.0e-2)
#     assert array.in_2d[4, 4] == pytest.approx(21059.12, 1.0e-2)
