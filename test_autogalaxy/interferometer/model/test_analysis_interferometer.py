from os import path

import autofit as af
import autogalaxy as ag

from autogalaxy.interferometer.model.result import ResultInterferometer


directory = path.dirname(path.realpath(__file__))


class TestAnalysisInterferometer:
    def test__make_result__result_interferometer_is_returned(self, interferometer_7):

        model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

        analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

        search = ag.m.MockSearch(name="test_search")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, ResultInterferometer)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, interferometer_7
    ):

        galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=0.1))

        model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

        analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        plane = analysis.plane_via_instance_from(instance=instance)

        fit = ag.FitInterferometer(dataset=interferometer_7, plane=plane)

        assert fit.log_likelihood == fit_figure_of_merit

    def test__fit_figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, interferometer_7
    ):
        hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=0.1))

        model = af.Collection(
            hyper_background_noise=hyper_background_noise,
            galaxies=af.Collection(galaxy=galaxy),
        )

        analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        plane = analysis.plane_via_instance_from(instance=instance)
        fit = ag.FitInterferometer(
            dataset=interferometer_7,
            plane=plane,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == fit_figure_of_merit
