from os import path
import os
import pytest
import shutil

from autoconf import conf
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


def clean(database_file, result_path):

    if path.exists(database_file):
        os.remove(database_file)

    if path.exists(result_path):
        shutil.rmtree(result_path)


def test__plane_generator_from_aggregator(masked_imaging_7x7, samples, model):

    path_prefix = "aggregator_plane_gen"

    database_file = path.join(conf.instance.output_path, "plane.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = mock.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)
    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)
    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    plane_gen = ag.agg.Plane(aggregator=agg)

    for plane in plane_gen:

        assert plane.galaxies[0].redshift == 0.5
        assert plane.galaxies[0].light.centre == (0.0, 1.0)
        assert plane.galaxies[1].redshift == 1.0

    clean(database_file=database_file, result_path=result_path)


#
def test__imaging_generator_from_aggregator(imaging_7x7, mask_2d_7x7, samples, model):

    path_prefix = "aggregator_imaging_gen"

    database_file = path.join(conf.instance.output_path, "imaging.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

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
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    imaging_gen = ag.agg.Imaging(aggregator=agg)

    for imaging in imaging_gen:
        assert (imaging.image == masked_imaging_7x7.image).all()
        assert isinstance(imaging.grid, ag.Grid2DIterate)
        assert isinstance(imaging.grid_inversion, ag.Grid2DIterate)
        assert imaging.grid.sub_steps == [2]
        assert imaging.grid.fractional_accuracy == 0.5

    clean(database_file=database_file, result_path=result_path)


def test__fit_imaging_generator_from_aggregator(masked_imaging_7x7, samples, model):

    path_prefix = "aggregator_fit_imaging_gen"

    database_file = path.join(conf.instance.output_path, "fit_imaging.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = mock.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    fit_imaging_gen = ag.agg.FitImaging(aggregator=agg)

    for fit_imaging in fit_imaging_gen:
        assert (fit_imaging.image == masked_imaging_7x7.image).all()

    clean(database_file=database_file, result_path=result_path)


def test__interferometer_generator_from_aggregator(
    visibilities_7,
    visibilities_noise_map_7,
    uv_wavelengths_7x2,
    mask_2d_7x7,
    samples,
    model,
):

    path_prefix = "aggregator_interferometer"

    database_file = path.join(conf.instance.output_path, "interferometer.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

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
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    interferometer_gen = ag.agg.Interferometer(aggregator=agg)

    for interferometer in interferometer_gen:
        assert (interferometer.visibilities == interferometer_7.visibilities).all()
        assert (interferometer.real_space_mask == mask_2d_7x7).all()
        assert isinstance(interferometer.grid, ag.Grid2DIterate)
        assert isinstance(interferometer.grid_inversion, ag.Grid2DIterate)
        assert interferometer.grid.sub_steps == [2]
        assert interferometer.grid.fractional_accuracy == 0.5
        assert isinstance(interferometer.transformer, ag.TransformerDFT)

    clean(database_file=database_file, result_path=result_path)


def test__fit_interferometer_generator_from_aggregator(
    interferometer_7, mask_2d_7x7, samples, model
):

    path_prefix = "aggregator_fit_interferometer"

    database_file = path.join(conf.instance.output_path, "fit_interferometer.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = mock.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    fit_interferometer_gen = ag.agg.FitInterferometer(aggregator=agg)

    for fit_interferometer in fit_interferometer_gen:
        assert (
            fit_interferometer.interferometer.visibilities
            == interferometer_7.visibilities
        ).all()
        assert (fit_interferometer.interferometer.real_space_mask == mask_2d_7x7).all()

    clean(database_file=database_file, result_path=result_path)
