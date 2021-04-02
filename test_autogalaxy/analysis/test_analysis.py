import pytest
import numpy as np
from os import path

import autofit as af
import autogalaxy as ag
from autogalaxy.analysis import result as res
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

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

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, hyper_result=result)

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


class TestAnalysisImaging:
    def test__make_result__result_imaging_is_returned(self, masked_imaging_7x7):

        model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

        search = mock.MockSearch(name="test_search")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultImaging)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, masked_imaging_7x7
    ):
        galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=0.1))

        model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        plane = analysis.plane_for_instance(instance=instance)

        fit = ag.FitImaging(imaging=masked_imaging_7x7, plane=plane)

        assert fit.log_likelihood == fit_figure_of_merit

    def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, masked_imaging_7x7
    ):

        hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=0.1))

        model = af.Collection(
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            galaxies=af.Collection(galaxy=galaxy),
        )

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        plane = analysis.plane_for_instance(instance=instance)
        fit = ag.FitImaging(
            imaging=masked_imaging_7x7,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == fit_figure_of_merit

    def test__uses_hyper_fit_correctly(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.galaxy = ag.Galaxy(
            redshift=0.5, light=ag.lp.EllSersic(intensity=1.0), mass=ag.mp.SphIsothermal
        )
        galaxies.source = ag.Galaxy(redshift=1.0, light=ag.lp.EllSersic())

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        galaxy_hyper_image = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
        galaxy_hyper_image[4] = 10.0
        hyper_model_image = ag.Array2D.full(
            fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {("galaxies", "galaxy"): galaxy_hyper_image}

        result = mock.MockResult(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7, hyper_result=result)

        hyper_galaxy = ag.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        )

        instance.galaxies.galaxy.hyper_galaxy = hyper_galaxy

        fit_likelihood = analysis.log_likelihood_function(instance=instance)

        g0 = ag.Galaxy(
            redshift=0.5,
            light_profile=instance.galaxies.galaxy.light,
            mass_profile=instance.galaxies.galaxy.mass,
            hyper_galaxy=hyper_galaxy,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_image=galaxy_hyper_image,
            hyper_minimum_value=0.0,
        )
        g1 = ag.Galaxy(redshift=1.0, light_profile=instance.galaxies.source.light)

        plane = ag.Plane(galaxies=[g0, g1])

        fit = ag.FitImaging(imaging=masked_imaging_7x7, plane=plane)

        assert (fit.plane.galaxies[0].hyper_galaxy_image == galaxy_hyper_image).all()
        assert fit_likelihood == fit.log_likelihood


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
