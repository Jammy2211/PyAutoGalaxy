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
            lens=af.Model(ag.Galaxy, redshift=0.5, light=ag.lp.EllSersic),
            source=af.Model(ag.Galaxy, redshift=1.0, light=ag.lp.EllSersic),
        )
    )


@pytest.fixture(name="samples")
def make_samples(model):
    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(centre=(0.0, 1.0)))
    galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.EllSersic())

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    parameters = [model.prior_count * [1.0], model.prior_count * [10.0]]

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


class MockResult:
    def __init__(self, log_likelihood):
        self.log_likelihood = log_likelihood
        self.log_evidence_values = log_likelihood
        self.model = log_likelihood


class MockAggregator:
    def __init__(self, grid_search_result):
        self.grid_search_result = grid_search_result

    @property
    def grid_search_results(self):
        return iter([self.grid_search_result])

    def values(self, str):
        return self.grid_search_results


# def test__plane_gen_from(masked_imaging_7x7, samples, model):
#
#     path_prefix = "aggregator_plane_gen"
#
#     database_file = path.join(conf.instance.output_path, "plane.sqlite")
#     result_path = path.join(conf.instance.output_path, path_prefix)
#
#     clean(database_file=database_file, result_path=result_path)
#
#     search = ag.m.MockSearch(samples=samples)
#     search.paths = af.DirectoryPaths(path_prefix=path_prefix)
#     analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)
#     search.fit(model=model, analysis=analysis)
#
#     agg = af.Aggregator.from_database(filename=database_file)
#     agg.add_directory(directory=result_path)
#
#     plane_gen = ag.agg.Plane(aggregator=agg)
#
#     for plane in plane_gen:
#
#         assert plane.galaxies[0].redshift == 0.5
#         assert plane.galaxies[0].light.centre == (0.0, 1.0)
#         assert plane.galaxies[1].redshift == 1.0
#
#     clean(database_file=database_file, result_path=result_path)


def test__plane_randomly_drawn_via_pdf_gen_from(masked_imaging_7x7, samples, model):

    path_prefix = "aggregator_plane_gen"

    database_file = path.join(conf.instance.output_path, "plane.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = ag.m.MockSearch(
        samples=samples, result=ag.m.MockResult(model=model, samples=samples)
    )
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)
    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)
    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    plane_agg = ag.agg.PlaneAgg(aggregator=agg)
    plane_pdf_gen = plane_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for plane_gen in plane_pdf_gen:

        for plane in plane_gen:

            i += 1

            assert plane.galaxies[0].redshift == 0.5
            assert plane.galaxies[0].light.centre == (10.0, 10.0)
            assert plane.galaxies[1].redshift == 1.0

    assert i == 2

    clean(database_file=database_file, result_path=result_path)


def test__plane_all_above_weight_gen(masked_imaging_7x7, samples, model):

    path_prefix = "aggregator_plane_gen"

    database_file = path.join(conf.instance.output_path, "plane.sqlite")
    result_path = path.join(conf.instance.output_path, path_prefix)

    clean(database_file=database_file, result_path=result_path)

    search = ag.m.MockSearch(
        samples=samples, result=ag.m.MockResult(model=model, samples=samples)
    )
    search.paths = af.DirectoryPaths(path_prefix=path_prefix)
    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)
    search.fit(model=model, analysis=analysis)

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=result_path)

    plane_agg = ag.agg.PlaneAgg(aggregator=agg)
    plane_pdf_gen = plane_agg.all_above_weight_gen_from(minimum_weight=-1.0)
    weight_pdf_gen = plane_agg.weights_above_gen_from(minimum_weight=-1.0)

    i = 0

    for (plane_gen, weight_gen) in zip(plane_pdf_gen, weight_pdf_gen):

        for plane in plane_gen:

            i += 1

            if i == 1:

                assert plane.galaxies[0].redshift == 0.5
                assert plane.galaxies[0].light.centre == (1.0, 1.0)
                assert plane.galaxies[1].redshift == 1.0

            if i == 2:

                assert plane.galaxies[0].redshift == 0.5
                assert plane.galaxies[0].light.centre == (10.0, 10.0)
                assert plane.galaxies[1].redshift == 1.0

        for weight in weight_gen:

            if i == 0:

                assert weight == 0.0

            if i == 1:

                assert weight == 1.0

    assert i == 2

    clean(database_file=database_file, result_path=result_path)


# def test__results_array_froms_file(path):
#
#     results = [
#         MockResult(log_likelihood=1.0),
#         MockResult(log_likelihood=2.0),
#         MockResult(log_likelihood=3.0),
#         MockResult(log_likelihood=4.0),
#     ]
#
#     lower_limit_lists = [[0.0, 0.0], [0.0, 0.5], [0.5, 0.0], [0.5, 0.5]]
#     physical_lower_limits_lists = [[-1.0, -1.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 0.0]]
#
#     grid_search_result = af.GridSearchResult(
#         results=results,
#         physical_lower_limits_lists=physical_lower_limits_lists,
#         lower_limit_lists=lower_limit_lists,
#     )
#
#     aggregator = MockAggregator(grid_search_result=grid_search_result)
#
#     array = ag.agg.grid_search_result_as_array(aggregator=aggregator)
#
#     assert array.native == pytest.approx(np.array([[3.0, 2.0], [1.0, 4.0]]), 1.0e4)
#     assert array.pixel_scales == (1.0, 1.0)


# def test__results_array_from_real_grid_search_pickle(path):
#
#     with open("{}/{}.pickle".format(path, "grid_search_result"), "rb") as f:
#         grid_search_result = pickle.load(f)
#
#     array = ag.agg.grid_search_log_evidences_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array.native)
#
#     array = ag.agg.grid_search_subhalo_masses_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array.native)
#
#     array = ag.agg.grid_search_subhalo_centres_as_array_from_grid_search_result(
#         grid_search_result=grid_search_result
#     )
#
#     print(array)
