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


@pytest.fixture(name="model")
def make_model():
    return af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
            source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
        )
    )


def test__dataset_generator_from_aggregator(masked_imaging_7x7, samples, model):

    search = mock.MockSearch(
        samples=samples, paths=af.Paths(path_prefix="aggregator_dataset_gen")
    )

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    dataset = list(agg.values("dataset"))


def test__plane_generator_from_aggregator(masked_imaging_7x7, samples, model):

    search = mock.MockSearch(
        samples=samples, paths=af.Paths(path_prefix="aggregator_plane_gen")
    )

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    plane_gen = ag.agg.Plane(aggregator=agg)

    for plane in plane_gen:

        assert plane.galaxies[0].redshift == 0.5
        assert plane.galaxies[0].light.centre == (0.0, 1.0)
        assert plane.galaxies[1].redshift == 1.0


def test__masked_imaging_generator_from_aggregator(
    imaging_7x7, mask_7x7, samples, model
):

    masked_imaging_7x7 = ag.MaskedImaging(
        imaging=imaging_7x7,
        mask=mask_7x7,
        settings=ag.SettingsMaskedImaging(
            grid_class=ag.Grid2DIterate,
            grid_inversion_class=ag.Grid2DIterate,
            fractional_accuracy=0.5,
            sub_steps=[2],
        ),
    )

    search = mock.MockSearch(
        samples=samples, paths=af.Paths(path_prefix="aggregator_masked_imaging_gen")
    )

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    masked_imaging_gen = ag.agg.MaskedImaging(aggregator=agg)

    for masked_imaging in masked_imaging_gen:
        assert (masked_imaging.imaging.image == imaging_7x7.image).all()
        assert isinstance(masked_imaging.grid, ag.Grid2DIterate)
        assert isinstance(masked_imaging.grid_inversion, ag.Grid2DIterate)
        assert masked_imaging.grid.sub_steps == [2]
        assert masked_imaging.grid.fractional_accuracy == 0.5


def test__fit_imaging_generator_from_aggregator(masked_imaging_7x7, samples, model):

    search = mock.MockSearch(
        samples=samples, paths=af.Paths(path_prefix="aggregator_fit_imaging_gen")
    )

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    fit_imaging_gen = ag.agg.FitImaging(aggregator=agg)

    for fit_imaging in fit_imaging_gen:
        assert (
            fit_imaging.masked_imaging.imaging.image == masked_imaging_7x7.imaging.image
        ).all()


def test__masked_interferometer_generator_from_aggregator(
    interferometer_7, visibilities_mask_7, mask_7x7, samples, model
):

    masked_interferometer = ag.MaskedInterferometer(
        interferometer=interferometer_7,
        visibilities_mask=visibilities_mask_7,
        real_space_mask=mask_7x7,
        settings=ag.SettingsMaskedInterferometer(
            grid_class=ag.Grid2DIterate,
            grid_inversion_class=ag.Grid2DIterate,
            fractional_accuracy=0.5,
            sub_steps=[2],
            transformer_class=ag.TransformerDFT,
        ),
    )

    search = mock.MockSearch(
        samples=samples, paths=af.Paths(path_prefix="aggregator_masked_interferometer")
    )

    analysis = ag.AnalysisInterferometer(dataset=masked_interferometer)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

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
    masked_interferometer_7, mask_7x7, samples, model
):

    search = mock.MockSearch(
        samples=samples, paths=af.Paths(path_prefix="aggregator_fit_interferometer_gen")
    )

    analysis = ag.AnalysisInterferometer(dataset=masked_interferometer_7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    fit_interferometer_gen = ag.agg.FitInterferometer(aggregator=agg)

    for fit_interferometer in fit_interferometer_gen:
        assert (
            fit_interferometer.masked_interferometer.interferometer.visibilities
            == masked_interferometer_7.interferometer.visibilities
        ).all()
        assert (
            fit_interferometer.masked_interferometer.real_space_mask == mask_7x7
        ).all()
