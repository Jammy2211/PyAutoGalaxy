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

    search = ag.m.MockSearch(samples=samples)
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    imaging_agg = ag.agg.ImagingAgg(aggregator=agg)
    imaging_gen = imaging_agg.imaging_gen_from()

    for imaging in imaging_gen:
        assert (imaging.image == masked_imaging_7x7.image).all()
        assert isinstance(imaging.grid, ag.Grid2DIterate)
        assert isinstance(imaging.grid_inversion, ag.Grid2DIterate)
        assert imaging.grid.sub_steps == [2]
        assert imaging.grid.fractional_accuracy == 0.5

    clean(database_file=database_file, result_path=result_path)
