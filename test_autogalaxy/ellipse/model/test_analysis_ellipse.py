from os import path

import autofit as af
import autogalaxy as ag

from autogalaxy.ellipse.model.result import ResultEllipse

directory = path.dirname(path.realpath(__file__))


def test__make_result__result_imaging_is_returned(masked_imaging_7x7):

    ellipse_list = af.Collection(
        af.Model(ag.Ellipse) for _ in range(2)
    )

    ellipse_list[0].major_axis = 1.0
    ellipse_list[1].major_axis = 2.0

    model = af.Collection(ellipses=ellipse_list)

    analysis = ag.AnalysisEllipse(dataset=masked_imaging_7x7)

    search = ag.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultEllipse)


# def test__figure_of_merit(
#     masked_imaging_7x7,
# ):
#     ellipse_list = [af.Model(ag.Ellipse(major_axis=1.0)), af.Model(ag.Ellipse(major_axis=2.0))]
#
#     model = af.Collection(ellipses=ellipse_list)
#
#     analysis = ag.AnalysisEllipse(dataset=masked_imaging_7x7)
#
#     instance = model.instance_from_unit_vector([])
#     fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)
#
#     fit_list = [ag.FitEllipse(dataset=masked_imaging_7x7, ellipse=ellipse) for ellipse in instance.ellipses]
#
#     assert fit_list[0].log_likelihood + fit_list[1].log_likelihood == fit_figure_of_merit
