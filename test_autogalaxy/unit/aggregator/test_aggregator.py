from os import path

import pytest

import autofit as af
import autogalaxy as ag
from autogalaxy.mock import mock

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="path")
def make_path():
    return path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


@pytest.fixture(name="samples")
def make_samples():
    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(centre=(0.0, 1.0)))
    galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.EllipticalSersic())

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    return mock.MockSamples(max_log_likelihood_instance=plane)


def test__dataset_generator_from_aggregator(imaging_7x7, mask_7x7, samples):
    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        search=mock.MockSearch(samples=samples, name="test_phase_aggregator"),
    )

    imaging_7x7.positions = ag.Grid2DIrregularGrouped([[1.0, 1.0], [2.0, 2.0]])

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults(samples=samples)
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    dataset = list(agg.values("dataset"))

    print(dataset)


def test__plane_generator_from_aggregator(imaging_7x7, mask_7x7, samples):
    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        search=mock.MockSearch(samples=samples, name="test_phase_aggregator"),
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
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        settings=ag.SettingsPhaseImaging(
            settings_masked_imaging=ag.SettingsMaskedImaging(
                grid_class=ag.Grid2DIterate,
                grid_inversion_class=ag.Grid2DIterate,
                fractional_accuracy=0.5,
                sub_steps=[2],
            )
        ),
        search=mock.MockSearch(samples=samples, name="test_phase_aggregator"),
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults(samples=samples)
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    masked_imaging_gen = ag.agg.MaskedImaging(aggregator=agg)

    for masked_imaging in masked_imaging_gen:
        assert (masked_imaging.imaging.image == imaging_7x7.image).all()
        assert isinstance(masked_imaging.grid, ag.Grid2DIterate)
        assert isinstance(masked_imaging.grid_inversion, ag.Grid2DIterate)
        assert masked_imaging.grid.sub_steps == [2]
        assert masked_imaging.grid.fractional_accuracy == 0.5


def test__fit_imaging_generator_from_aggregator(imaging_7x7, mask_7x7, samples):
    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        search=mock.MockSearch(samples=samples, name="test_phase_aggregator"),
    )

    phase_imaging_7x7.run(
        dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults(samples=samples)
    )

    agg = af.Aggregator(directory=phase_imaging_7x7.paths.output_path)

    fit_imaging_gen = ag.agg.FitImaging(aggregator=agg)

    for fit_imaging in fit_imaging_gen:
        assert (fit_imaging.masked_imaging.imaging.image == imaging_7x7.image).all()


def test__masked_interferometer_generator_from_aggregator(
    interferometer_7, visibilities_mask_7, mask_7x7, samples
):
    phase_interferometer_7x7 = ag.PhaseInterferometer(
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        settings=ag.SettingsPhaseInterferometer(
            settings_masked_interferometer=ag.SettingsMaskedInterferometer(
                grid_class=ag.Grid2DIterate,
                grid_inversion_class=ag.Grid2DIterate,
                fractional_accuracy=0.5,
                sub_steps=[2],
                transformer_class=ag.TransformerDFT,
            )
        ),
        search=mock.MockSearch(samples=samples, name="test_phase_aggregator"),
        real_space_mask=mask_7x7,
    )

    phase_interferometer_7x7.run(
        dataset=interferometer_7,
        mask=visibilities_mask_7,
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
        assert isinstance(masked_interferometer.grid, ag.Grid2DIterate)
        assert isinstance(masked_interferometer.grid_inversion, ag.Grid2DIterate)
        assert masked_interferometer.grid.sub_steps == [2]
        assert masked_interferometer.grid.fractional_accuracy == 0.5
        assert isinstance(masked_interferometer.transformer, ag.TransformerDFT)


def test__fit_interferometer_generator_from_aggregator(
    interferometer_7, visibilities_mask_7, mask_7x7, samples
):
    phase_interferometer_7x7 = ag.PhaseInterferometer(
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        ),
        search=mock.MockSearch(samples=samples, name="test_phase_aggregator"),
        real_space_mask=mask_7x7,
    )

    phase_interferometer_7x7.run(
        dataset=interferometer_7,
        mask=visibilities_mask_7,
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
