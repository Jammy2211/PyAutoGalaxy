import autogalaxy as ag

from test_autogalaxy.aggregator.conftest import clean, aggregator_from

database_file = "db_galaxies"


def test__galaxies_randomly_drawn_via_pdf_gen_from(
    masked_imaging_7x7,
    samples,
    model,
):
    clean(database_file=database_file)

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model,
        samples=samples,
    )

    galaxies_agg = ag.agg.GalaxiesAgg(aggregator=agg)
    galaxies_pdf_gen = galaxies_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for galaxies_gen in galaxies_pdf_gen:
        for galaxy_list in galaxies_gen:
            i += 1

            assert galaxy_list[0].g0.redshift == 0.5
            assert galaxy_list[0].g0.light.centre == (10.0, 10.0)
            assert galaxy_list[0].g1.redshift == 1.0

    assert i == 2

    clean(database_file=database_file)


def test__galaxies_all_above_weight_gen(
    masked_imaging_7x7,
    samples,
    model,
):
    clean(database_file=database_file)

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, use_jax=False)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model,
        samples=samples,
    )

    galaxies_agg = ag.agg.GalaxiesAgg(aggregator=agg)
    galaxies_pdf_gen = galaxies_agg.all_above_weight_gen_from(minimum_weight=-1.0)
    weight_pdf_gen = galaxies_agg.weights_above_gen_from(minimum_weight=-1.0)

    i = 0

    for galaxies_gen, weight_gen in zip(galaxies_pdf_gen, weight_pdf_gen):
        for galaxy_list in galaxies_gen:
            i += 1

            if i == 1:
                assert galaxy_list[0].g0.redshift == 0.5
                assert galaxy_list[0].g0.light.centre == (1.0, 1.0)
                assert galaxy_list[0].g1.redshift == 1.0

            if i == 2:
                assert galaxy_list[0].g0.redshift == 0.5
                assert galaxy_list[0].g0.light.centre == (10.0, 10.0)
                assert galaxy_list[0].g1.redshift == 1.0

        for weight in weight_gen:
            if i == 0:
                assert weight == 0.0

            if i == 1:
                assert weight == 1.0

    assert i == 2

    clean(database_file=database_file)
