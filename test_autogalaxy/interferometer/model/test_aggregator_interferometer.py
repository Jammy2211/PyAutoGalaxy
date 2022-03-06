from os import path
import os
import pytest
import shutil

from autoconf import conf
import autofit as af
import autogalaxy as ag

from autofit.non_linear.samples import Sample

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="path")
def make_path():
    return path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


@pytest.fixture(name="model")
def make_model():
    return af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, light=ag.lp.EllSersic),
            source=af.Model(ag.Galaxy, redshift=1.0, light=ag.lp.EllSersic),
        )
    )


@pytest.fixture(name="samples")
def make_samples(model):
    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(centre=(0.0, 1.0)))
    galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.EllSersic())

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    parameters = [model.prior_count * [0.0], model.prior_count * [10.0]]

    sample_list = Sample.from_lists(
        model=model,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )

    return ag.m.MockSamples(
        model=model, sample_list=sample_list, max_log_likelihood_instance=plane
    )


def clean(database_file, result_path):

    if path.exists(database_file):
        os.remove(database_file)

    if path.exists(result_path):
        shutil.rmtree(result_path)


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

    search = ag.m.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    interferometer_agg = ag.agg.InterferometerAgg(aggregator=agg)
    interferometer_gen = interferometer_agg.interferometer_gen_from()

    for interferometer in interferometer_gen:
        assert (interferometer.visibilities == interferometer_7.visibilities).all()
        assert (interferometer.real_space_mask == mask_2d_7x7).all()
        assert isinstance(interferometer.grid, ag.Grid2DIterate)
        assert isinstance(interferometer.grid_inversion, ag.Grid2DIterate)
        assert interferometer.grid.sub_steps == [2]
        assert interferometer.grid.fractional_accuracy == 0.5
        assert isinstance(interferometer.transformer, ag.TransformerDFT)

    clean(database_file=database_file, result_path=result_path)
