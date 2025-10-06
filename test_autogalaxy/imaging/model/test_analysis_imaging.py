from os import path

import autofit as af
import autogalaxy as ag

from autogalaxy.imaging.model.result import ResultImaging

directory = path.dirname(path.realpath(__file__))


def test__make_result__result_imaging_is_returned(masked_imaging_7x7):
    model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    search = ag.m.MockSearch(name="test_search")

    result = search.fit(model=model, analysis=analysis)

    assert isinstance(result, ResultImaging)


def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
    masked_imaging_7x7,
):
    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

    analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

    instance = model.instance_from_unit_vector([])
    fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

    galaxies = analysis.galaxies_via_instance_from(instance=instance)

    fit = ag.FitImaging(dataset=masked_imaging_7x7, galaxies=galaxies)

    assert fit.log_likelihood == fit_figure_of_merit
