from os import path

import autofit as af
import autogalaxy as ag

directory = path.dirname(path.realpath(__file__))


def test__fit_figure_of_merit__includes_hyper_image_and_noise__matches_fit(
    interferometer_7,
):
    hyper_background_noise = ag.legacy.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

    galaxy = ag.legacy.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=0.1))

    model = af.Collection(
        hyper_background_noise=hyper_background_noise,
        galaxies=af.Collection(galaxy=galaxy),
    )

    analysis = ag.legacy.AnalysisInterferometer(dataset=interferometer_7)

    instance = model.instance_from_unit_vector([])
    fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

    plane = analysis.plane_via_instance_from(instance=instance)
    fit = ag.legacy.FitInterferometer(
        dataset=interferometer_7,
        plane=plane,
        hyper_background_noise=hyper_background_noise,
    )

    assert fit.log_likelihood == fit_figure_of_merit
