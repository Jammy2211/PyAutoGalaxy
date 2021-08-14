import pytest
import numpy as np
from os import path

import autofit as af
import autogalaxy as ag
from autogalaxy.analysis import result as res
from autogalaxy.mock import mock


directory = path.dirname(path.realpath(__file__))


class TestAnalysisDataset:
    def test__associate_hyper_images(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.galaxy = ag.Galaxy(redshift=0.5)
        galaxies.source = ag.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        hyper_galaxy_image_path_dict = {
            ("galaxies", "galaxy"): ag.Array2D.ones(
                shape_native=(3, 3), pixel_scales=1.0
            ),
            ("galaxies", "source"): ag.Array2D.full(
                fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
            ),
        }

        result = mock.MockResult(
            instance=instance,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=ag.Array2D.full(
                fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
            ),
        )

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7, hyper_dataset_result=result
        )

        instance = analysis.associate_hyper_images(instance=instance)

        assert instance.galaxies.galaxy.hyper_galaxy_image.native == pytest.approx(
            np.ones((3, 3)), 1.0e-4
        )
        assert instance.galaxies.source.hyper_galaxy_image.native == pytest.approx(
            2.0 * np.ones((3, 3)), 1.0e-4
        )

        assert instance.galaxies.galaxy.hyper_model_image.native == pytest.approx(
            3.0 * np.ones((3, 3)), 1.0e-4
        )
        assert instance.galaxies.source.hyper_model_image.native == pytest.approx(
            3.0 * np.ones((3, 3)), 1.0e-4
        )


class TestAnalysisInterferometer:
    def test__make_result__result_interferometer_is_returned(self, interferometer_7):

        model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

        analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

        search = mock.MockSearch(name="test_search")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultInterferometer)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, interferometer_7
    ):

        galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=0.1))

        model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

        analysis = ag.AnalysisInterferometer(dataset=interferometer_7)

        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        plane = analysis.plane_for_instance(instance=instance)

        fit = ag.FitInterferometer(interferometer=interferometer_7, plane=plane)

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

        plane = analysis.plane_for_instance(instance=instance)
        fit = ag.FitInterferometer(
            interferometer=interferometer_7,
            plane=plane,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == fit_figure_of_merit
