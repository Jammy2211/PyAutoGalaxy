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
    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(centre=(0.0, 1.0)))
    galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.EllSersic())

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    return mock.MockSamples(max_log_likelihood_instance=plane)


@pytest.fixture(name="model")
def make_model():
    return af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, light=ag.lp.EllSersic),
            source=af.Model(ag.Galaxy, redshift=1.0, light=ag.lp.EllSersic),
        )
    )


def test__dataset_generator_from_aggregator(masked_imaging_7x7, samples, model):

    search = mock.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix="aggregator_dataset_gen")

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    dataset = list(agg.values("dataset"))


def test__plane_generator_from_aggregator(masked_imaging_7x7, samples, model):

    search = mock.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix="aggregator_plane_gen")

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    plane_gen = ag.agg.Plane(aggregator=agg)

    for plane in plane_gen:

        assert plane.galaxies[0].redshift == 0.5
        assert plane.galaxies[0].light.centre == (0.0, 1.0)
        assert plane.galaxies[1].redshift == 1.0


def test__imaging_generator_from_aggregator(imaging_7x7, mask_2d_7x7, samples, model):

    masked_imaging_7x7 = imaging_7x7.apply_mask(mask=mask_2d_7x7)

    masked_imaging_7x7 = masked_imaging_7x7.apply_settings(
        settings=ag.SettingsImaging(
            grid_class=ag.Grid2DIterate,
            grid_inversion_class=ag.Grid2DIterate,
            fractional_accuracy=0.5,
            sub_steps=[2],
        )
    )

    search = mock.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix="aggregator_masked_imaging_gen")

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    imaging_gen = ag.agg.Imaging(aggregator=agg)

    for imaging in imaging_gen:
        assert (imaging.image == masked_imaging_7x7.image).all()
        assert isinstance(imaging.grid, ag.Grid2DIterate)
        assert isinstance(imaging.grid_inversion, ag.Grid2DIterate)
        assert imaging.grid.sub_steps == [2]
        assert imaging.grid.fractional_accuracy == 0.5


def test__fit_imaging_generator_from_aggregator(masked_imaging_7x7, samples, model):

    search = mock.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix="aggregator_fit_imaging_gen")

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    fit_imaging_gen = ag.agg.FitImaging(aggregator=agg)

    for fit_imaging in fit_imaging_gen:
        assert (fit_imaging.image == masked_imaging_7x7.image).all()


def test__interferometer_generator_from_aggregator(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
    samples,
    model,
):
    interferometer_7 = ag.Interferometer(
        visibilities=visibilities_7,
        noise_map=visibilities_noise_map_7,
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        settings=ag.SettingsInterferometer(
            grid_class=ag.Grid2DIterate,
            grid_inversion_class=ag.Grid2DIterate,
            fractional_accuracy=0.5,
            sub_steps=[2],
            transformer_class=ag.TransformerDFT,
        ),
    )

    search = mock.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix="aggregator_interferometer")

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    interferometer_gen = ag.agg.Interferometer(aggregator=agg)

    for interferometer in interferometer_gen:
        assert (interferometer.visibilities == interferometer_7.visibilities).all()
        assert (interferometer.real_space_mask == mask_2d_7x7).all()
        assert isinstance(interferometer.grid, ag.Grid2DIterate)
        assert isinstance(interferometer.grid_inversion, ag.Grid2DIterate)
        assert interferometer.grid.sub_steps == [2]
        assert interferometer.grid.fractional_accuracy == 0.5
        assert isinstance(interferometer.transformer, ag.TransformerDFT)


def test__fit_interferometer_generator_from_aggregator(
    interferometer_7, mask_2d_7x7, samples, model
):

    search = mock.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix="aggregator_fit_interferometer_gen")

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator(directory=search.paths.output_path)

    fit_interferometer_gen = ag.agg.FitInterferometer(aggregator=agg)

    for fit_interferometer in fit_interferometer_gen:
        assert (
            fit_interferometer.interferometer.visibilities
            == interferometer_7.visibilities
        ).all()
        assert (fit_interferometer.interferometer.real_space_mask == mask_2d_7x7).all()
