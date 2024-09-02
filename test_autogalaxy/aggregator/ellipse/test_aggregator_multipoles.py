import autogalaxy as ag

from test_autogalaxy.aggregator.conftest import clean, aggregator_from

database_file = "db_multipoles"


def test__multipoles_randomly_drawn_via_pdf_gen_from(
    masked_imaging_7x7,
    samples,
    model,
):
    clean(database_file=database_file)

    analysis = ag.AnalysisEllipse(dataset=masked_imaging_7x7)

    agg = aggregator_from(
        database_file=database_file,
        analysis=analysis,
        model=model,
        samples=samples,
    )

    multipoles_agg = ag.agg.MultipolesAgg(aggregator=agg)
    multipoles_pdf_gen = multipoles_agg.randomly_drawn_via_pdf_gen_from(total_samples=2)

    i = 0

    for multipoles_gen in multipoles_pdf_gen:
        for multipoles_lists_list in multipoles_gen:

            multipole_list = multipoles_lists_list[0]

            i += 1

            assert multipole_list[0][0].m == 1
            assert multipole_list[0][1].m == 2
            assert multipole_list[1][0].m == 1
            assert multipole_list[1][1].m == 2

    assert i == 2

    clean(database_file=database_file)
#
#
# def test__multipoles_all_above_weight_gen(
#     masked_imaging_7x7,
#     samples,
#     model,
# ):
#     clean(database_file=database_file)
#
#     analysis = ag.AnalysisEllipse(dataset=masked_imaging_7x7)
#
#     agg = aggregator_from(
#         database_file=database_file,
#         analysis=analysis,
#         model=model,
#         samples=samples,
#     )
#
#     multipoles_agg = ag.agg.MultipolesAgg(aggregator=agg)
#     multipoles_pdf_gen = multipoles_agg.all_above_weight_gen_from(minimum_weight=-1.0)
#     weight_pdf_gen = multipoles_agg.weights_above_gen_from(minimum_weight=-1.0)
#
#     i = 0
#
#     for multipoles_gen, weight_gen in zip(multipoles_pdf_gen, weight_pdf_gen):
#         for multipoles_lists_list in multipoles_gen:
#
#             multipole_list = multipoles_lists_list[0]
#
#             i += 1
#
#             if i == 1:
#                 assert multipole_list[0].centre == (1.0, 1.0)
#             else:
#                 assert multipole_list[0].centre == (10.0, 10.0)
#
#             assert multipole_list[0].major_axis == 0
#             assert multipole_list[1].major_axis == 1
#
#         for weight in weight_gen:
#             if i == 0:
#                 assert weight == 0.0
#
#             if i == 1:
#                 assert weight == 1.0
#
#     assert i == 2
#
#     clean(database_file=database_file)
