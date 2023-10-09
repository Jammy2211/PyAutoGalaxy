import autogalaxy as ag

from test_autogalaxy.aggregator.conftest import clean, aggregator_from

database_file = "db_plane"


def test__plane_randomly_drawn_via_pdf_gen_from(
    masked_imaging_7x7,
    adapt_model_image_7x7,
    adapt_galaxy_image_path_dict_7x7,
    samples,
    model,
):
    clean(database_file=database_file)

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    analysis.adapt_model_image = adapt_model_image_7x7
    analysis.adapt_galaxy_image_path_dict = adapt_galaxy_image_path_dict_7x7

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model,
        samples=samples,
    )

    plane_agg = ag.agg.PlaneAgg(aggregator=agg)
    plane_pdf_gen = plane_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for plane_gen in plane_pdf_gen:
        for plane_list in plane_gen:
            i += 1

            assert plane_list[0].galaxies[0].redshift == 0.5
            assert plane_list[0].galaxies[0].light.centre == (10.0, 10.0)
            assert plane_list[0].galaxies[1].redshift == 1.0

            assert (
                plane_list[0].galaxies[0].adapt_model_image == adapt_model_image_7x7
            ).all()
            assert (
                plane_list[0].galaxies[0].adapt_galaxy_image
                == adapt_galaxy_image_path_dict_7x7["('galaxies', 'g0')"]
            ).all()
            assert (
                plane_list[0].galaxies[1].adapt_galaxy_image
                == adapt_galaxy_image_path_dict_7x7["('galaxies', 'g1')"]
            ).all()

    assert i == 2

    clean(database_file=database_file)


def test__plane_all_above_weight_gen(
    masked_imaging_7x7,
    adapt_model_image_7x7,
    adapt_galaxy_image_path_dict_7x7,
    samples,
    model,
):
    clean(database_file=database_file)

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    analysis.adapt_model_image = adapt_model_image_7x7
    analysis.adapt_galaxy_image_path_dict = adapt_galaxy_image_path_dict_7x7

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model,
        samples=samples,
    )

    plane_agg = ag.agg.PlaneAgg(aggregator=agg)
    plane_pdf_gen = plane_agg.all_above_weight_gen_from(minimum_weight=-1.0)
    weight_pdf_gen = plane_agg.weights_above_gen_from(minimum_weight=-1.0)

    i = 0

    for plane_gen, weight_gen in zip(plane_pdf_gen, weight_pdf_gen):
        for plane_list in plane_gen:
            i += 1

            if i == 1:
                assert plane_list[0].galaxies[0].redshift == 0.5
                assert plane_list[0].galaxies[0].light.centre == (1.0, 1.0)
                assert plane_list[0].galaxies[1].redshift == 1.0

                assert (
                    plane_list[0].galaxies[0].adapt_model_image == adapt_model_image_7x7
                ).all()
                assert (
                    plane_list[0].galaxies[0].adapt_galaxy_image
                    == adapt_galaxy_image_path_dict_7x7["('galaxies', 'g0')"]
                ).all()
                assert (
                    plane_list[0].galaxies[1].adapt_galaxy_image
                    == adapt_galaxy_image_path_dict_7x7["('galaxies', 'g1')"]
                ).all()

            if i == 2:
                assert plane_list[0].galaxies[0].redshift == 0.5
                assert plane_list[0].galaxies[0].light.centre == (10.0, 10.0)
                assert plane_list[0].galaxies[1].redshift == 1.0

        for weight in weight_gen:
            if i == 0:
                assert weight == 0.0

            if i == 1:
                assert weight == 1.0

    assert i == 2

    clean(database_file=database_file)
