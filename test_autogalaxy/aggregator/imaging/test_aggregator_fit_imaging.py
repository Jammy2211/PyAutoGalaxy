from pathlib import Path

import pytest

import autogalaxy as ag
import autofit as af
from uuid import uuid4
from autoconf import conf
from autoconf.conf import with_config

from test_autogalaxy.aggregator.conftest import clean, aggregator_from

database_file = "db_fit_imaging"


@pytest.fixture(name="agg_7x7")
@with_config(
    "general",
    "output",
    "samples_to_csv",
    value=True,
)
def make_agg_7x7(samples, model, analysis_imaging_7x7):
    output_path = Path(conf.instance.output_path)

    search = ag.m.MockSearch(
        samples=samples, result=ag.m.MockResult(model=model, samples=samples)
    )
    file_prefix = str(uuid4())
    search.paths = af.DirectoryPaths(path_prefix=file_prefix)
    search.fit(model=model, analysis=analysis_imaging_7x7)

    analysis_imaging_7x7.visualize_before_fit(paths=search.paths, model=model)

    database_file = output_path / f"{file_prefix}.sqlite"

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=output_path / file_prefix)
    return agg


def test__fit_imaging__max_log_likelihood__analysis_has_single_dataset(
    agg_7x7,
):
    fit_agg = ag.agg.FitImagingAgg(aggregator=agg_7x7)
    fit_max_lh_gen = fit_agg.max_log_likelihood_gen_from()

    for (
        fit_list
    ) in (
        fit_max_lh_gen
    ):  # Only Max LH sample so fit_list contains 1 lists of a single fit.

        assert fit_list[0].galaxies[0].redshift == 0.5
        assert fit_list[0].galaxies[0].light.centre == (10.0, 10.0)

        assert fit_list[0].dataset_model.background_sky_level == 10.0

    clean(database_file=database_file)


def test__fit_imaging__randomly_drawn_via_pdf_gen_from__analysis_has_single_dataset(
    agg_7x7,
):
    fit_agg = ag.agg.FitImagingAgg(aggregator=agg_7x7)
    fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=3)

    i = 0

    for fit_list_gen in fit_pdf_gen:  # 1 Dataset so just one fit
        for (
            fit_list
        ) in (
            fit_list_gen
        ):  # Iterate over each total_samples=3, each with two fits for each analysis.

            i += 1

            # Check fit for each `Analysis` so take first and only dataset.
            assert fit_list[0].galaxies[0].redshift == 0.5
            assert fit_list[0].galaxies[0].light.centre == (10.0, 10.0)
            assert fit_list[0].dataset_model.background_sky_level == 10.0

    assert i == 3

    clean(database_file=database_file)


# TODO : These need to use FactorGraphModel


# def test__fit_imaging__max_log_likelihood_gen_from__analysis_multi(
#     analysis_imaging_7x7, samples, model
# ):
#
#     analysis_factor_list = []
#
#     for i, analysis in enumerate([analysis_imaging_7x7, analysis_imaging_7x7]):
#
#         model_analysis = model.copy()
#         analysis_factor = af.AnalysisFactor(prior_model=model_analysis, analysis=analysis)
#
#         analysis_factor_list.append(analysis_factor)
#
#     factor_graph = af.FactorGraphModel(*analysis_factor_list)
#
#     agg = aggregator_from(
#         database_file=database_file,
#         analysis=factor_graph,
#         model=factor_graph.global_prior_model,
#         samples=samples,
#     )
#
#     fit_agg = ag.agg.FitImagingAgg(aggregator=agg)
#
#     fit_max_lh_gen = fit_agg.max_log_likelihood_gen_from()
#
#     for fit_list in fit_max_lh_gen: # Only Max LH sample so fit_list contains 1 lists of the 2 fit (one for each analysis).
#
#         assert fit_list[0].galaxies[0].redshift == 0.5
#         assert fit_list[0].galaxies[0].light.centre == (10.0, 10.0)
#
#         assert fit_list[1].galaxies[0].redshift == 0.5
#         assert fit_list[1].galaxies[0].light.centre == (10.0, 10.0)
#
#     clean(database_file=database_file)
#
# def test__fit_imaging__randomly_drawn_via_pdf_gen_from__analysis_multi(
#     analysis_imaging_7x7, samples, model
# ):
#     agg = aggregator_from(
#         database_file=database_file,
#         analysis=analysis_imaging_7x7 + analysis_imaging_7x7,
#         model=model,
#         samples=samples,
#     )
#
#     fit_agg = ag.agg.FitImagingAgg(aggregator=agg)
#     fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=3)
#
#     i = 0
#
#     for fit_gen in fit_pdf_gen: # 1 Dataset so just one fit
#         for fit_list in fit_gen: # Iterate over each total_samples=3, each with two fits for each analysis.
#             i += 1
#
#             assert fit_list[0].galaxies[0].redshift == 0.5
#             assert fit_list[0].galaxies[0].light.centre == (10.0, 10.0)
#
#             assert fit_list[1].galaxies[0].redshift == 0.5
#             assert fit_list[1].galaxies[0].light.centre == (10.0, 10.0)
#
#     assert i == 3
#
#     clean(database_file=database_file)


def test__fit_imaging_all_above_weight_gen(agg_7x7):
    fit_agg = ag.agg.FitImagingAgg(aggregator=agg_7x7)
    fit_pdf_gen = fit_agg.all_above_weight_gen_from(minimum_weight=-1.0)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1

            if i == 1:
                assert fit_list[0].galaxies[0].redshift == 0.5
                assert fit_list[0].galaxies[0].light.centre == (1.0, 1.0)

            if i == 2:
                assert fit_list[0].galaxies[0].redshift == 0.5
                assert fit_list[0].galaxies[0].light.centre == (10.0, 10.0)

    assert i == 2

    clean(database_file=database_file)


def test__fit_imaging__adapt_images(agg_7x7, adapt_images_7x7):
    fit_agg = ag.agg.FitImagingAgg(aggregator=agg_7x7)
    fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1

            assert (
                list(fit_list[0].adapt_images.galaxy_image_dict.values())[0]
                == list(adapt_images_7x7.galaxy_name_image_dict.values())[0]
            ).all()

    assert i == 2

    # clean(database_file=database_file)
