from pathlib import Path

import pytest

import autogalaxy as ag
import autofit as af
from uuid import uuid4
from autoconf import conf
from autoconf.conf import with_config

from test_autogalaxy.aggregator.conftest import clean, aggregator_from

database_file = "db_fit_ellipse"


@pytest.fixture(name="agg_7x7")
@with_config(
    "general",
    "output",
    "samples_to_csv",
    value=True,
)
def make_agg_7x7(samples, model, analysis_ellipse_7x7):
    output_path = Path(conf.instance.output_path)

    search = ag.m.MockSearch(
        samples=samples, result=ag.m.MockResult(model=model, samples=samples)
    )
    file_prefix = str(uuid4())
    search.paths = af.DirectoryPaths(path_prefix=file_prefix)
    search.fit(model=model, analysis=analysis_ellipse_7x7)

    analysis_ellipse_7x7.visualize_before_fit(paths=search.paths, model=model)

    database_file = output_path / f"{file_prefix}.sqlite"

    agg = af.Aggregator.from_database(filename=database_file)
    agg.add_directory(directory=output_path / file_prefix)
    return agg


def test__fit_ellipse_randomly_drawn_via_pdf_gen_from__analysis_has_single_dataset(
    agg_7x7,
):
    fit_agg = ag.agg.FitEllipseAgg(aggregator=agg_7x7)
    fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_lists_list in fit_gen:
            # Only one `Analysis` so take first and only dataset.
            fit_list = fit_lists_list[0]

            i += 1

            assert fit_list[0].ellipse.major_axis == 0
            assert fit_list[1].ellipse.major_axis == 1

            assert fit_list[0].multipole_list[0].m == 1
            assert fit_list[0].multipole_list[1].m == 2
            assert fit_list[1].multipole_list[0].m == 1
            assert fit_list[1].multipole_list[1].m == 2

    assert i == 2

    clean(database_file=database_file)


def test__fit_ellipse_all_above_weight_gen(agg_7x7):
    fit_agg = ag.agg.FitEllipseAgg(aggregator=agg_7x7)
    fit_pdf_gen = fit_agg.all_above_weight_gen_from(minimum_weight=-1.0)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_lists_list in fit_gen:
            # Only one `Analysis` so take first and only dataset.
            fit_list = fit_lists_list[0]

            i += 1

            if i == 1:
                assert fit_list[0].ellipse.centre == (1.0, 1.0)

            if i == 2:
                assert fit_list[0].ellipse.centre == (10.0, 10.0)

    assert i == 2

    clean(database_file=database_file)
