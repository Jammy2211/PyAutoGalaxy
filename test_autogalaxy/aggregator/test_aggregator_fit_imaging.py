
import autogalaxy as ag

from test_autogalaxy.aggregator.conftest import clean, aggregator_from

database_file = "db_fit_imaging"

def test__fit_imaging_randomly_drawn_via_pdf_gen_from__analysis_has_single_dataset(
    analysis_imaging_7x7, samples, model
):

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis_imaging_7x7,
        model=model,
        samples=samples,
    )

    fit_agg = ag.agg.FitImagingAgg(aggregator=agg)
    fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(
        total_samples=2
    )

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1

            assert fit_list[0].plane.galaxies[0].redshift == 0.5
            assert fit_list[0].plane.galaxies[0].light.centre == (10.0, 10.0)

    assert i == 2

    clean(database_file=database_file)


def test__fit_imaging_randomly_drawn_via_pdf_gen_from__analysis_multi(
    analysis_imaging_7x7, samples, model
):

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis_imaging_7x7 + analysis_imaging_7x7,
        model=model,
        samples=samples,
    )

    fit_agg = ag.agg.FitImagingAgg(aggregator=agg)
    fit_pdf_gen = fit_agg.randomly_drawn_via_pdf_gen_from(
        total_samples=2
    )

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1

            assert fit_list[0].plane.galaxies[0].redshift == 0.5
            assert fit_list[0].plane.galaxies[0].light.centre == (10.0, 10.0)

            assert fit_list[1].plane.galaxies[0].redshift == 0.5
            assert fit_list[1].plane.galaxies[0].light.centre == (10.0, 10.0)

    assert i == 2

    clean(database_file=database_file)


def test__fit_imaging_all_above_weight_gen(analysis_imaging_7x7, samples, model):

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis_imaging_7x7,
        model=model,
        samples=samples,
    )

    fit_agg = ag.agg.FitImagingAgg(aggregator=agg)
    fit_pdf_gen = fit_agg.all_above_weight_gen_from(minimum_weight=-1.0)

    i = 0

    for fit_gen in fit_pdf_gen:
        for fit_list in fit_gen:
            i += 1

            if i == 1:
                assert fit_list[0].plane.galaxies[0].redshift == 0.5
                assert fit_list[0].plane.galaxies[0].light.centre == (1.0, 1.0)

            if i == 2:
                assert fit_list[0].plane.galaxies[0].redshift == 0.5
                assert fit_list[0].plane.galaxies[0].light.centre == (10.0, 10.0)

    assert i == 2

    clean(database_file=database_file)
