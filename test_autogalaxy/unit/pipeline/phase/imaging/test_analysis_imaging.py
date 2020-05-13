from os import path

import autofit as af
import autogalaxy as ag
import pytest
from astropy import cosmology as cosmo
from autogalaxy.fit.fit import FitImaging
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestFit:
    def test__fit_using_imaging(self, imaging_7x7, mask_7x7):

        phase_imaging_7x7 = ag.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
                source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
            ),
            phase_name="test_phase_test_fit",
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        assert isinstance(result.instance.galaxies[0], ag.Galaxy)
        assert isinstance(result.instance.galaxies[0], ag.Galaxy)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, imaging_7x7, mask_7x7
    ):
        lens_galaxy = ag.Galaxy(
            redshift=0.5, light=ag.lp.EllipticalSersic(intensity=0.1)
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=[lens_galaxy],
            cosmology=cosmo.FLRW,
            sub_size=1,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        masked_imaging = ag.MaskedImaging(imaging=imaging_7x7, mask=mask_7x7)
        plane = analysis.plane_for_instance(instance=instance)

        fit = ag.FitImaging(masked_imaging=masked_imaging, plane=plane)

        assert fit.log_likelihood == fit_figure_of_merit

    def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, imaging_7x7, mask_7x7
    ):
        hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        lens_galaxy = ag.Galaxy(
            redshift=0.5, light=ag.lp.EllipticalSersic(intensity=0.1)
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=[lens_galaxy],
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            cosmology=cosmo.FLRW,
            sub_size=4,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        mask = phase_imaging_7x7.meta_dataset.mask_with_phase_sub_size_from_mask(
            mask=mask_7x7
        )
        assert mask.sub_size == 4

        masked_imaging = ag.MaskedImaging(imaging=imaging_7x7, mask=mask)
        plane = analysis.plane_for_instance(instance=instance)
        fit = FitImaging(
            masked_imaging=masked_imaging,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == fit_figure_of_merit

    def test__uses_hyper_fit_correctly(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.lens = ag.Galaxy(
            redshift=0.5,
            light=ag.lp.EllipticalSersic(intensity=1.0),
            mass=ag.mp.SphericalIsothermal,
        )
        galaxies.source = ag.Galaxy(redshift=1.0, light=ag.lp.EllipticalSersic())

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        lens_hyper_image = ag.Array.ones(shape_2d=(3, 3), pixel_scales=0.1)
        lens_hyper_image[4] = 10.0
        hyper_model_image = ag.Array.full(
            fill_value=0.5, shape_2d=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {("galaxies", "lens"): lens_hyper_image}

        results = mock_pipeline.MockResults(
            use_as_hyper_dataset=True,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        analysis = ag.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            cosmology=cosmo.Planck15,
            results=results,
            image_path="files/",
        )

        hyper_galaxy = ag.HyperGalaxy(
            contribution_factor=1.0, noise_factor=1.0, noise_power=1.0
        )

        instance.galaxies.lens.hyper_galaxy = hyper_galaxy

        fit_likelihood = analysis.log_likelihood_function(instance=instance)

        g0 = ag.Galaxy(
            redshift=0.5,
            light_profile=instance.galaxies.lens.light,
            mass_profile=instance.galaxies.lens.mass,
            hyper_galaxy=hyper_galaxy,
            hyper_model_image=hyper_model_image,
            hyper_galaxy_image=lens_hyper_image,
            hyper_minimum_value=0.0,
        )
        g1 = ag.Galaxy(redshift=1.0, light_profile=instance.galaxies.source.light)

        plane = ag.Plane(galaxies=[g0, g1])

        fit = FitImaging(masked_imaging=masked_imaging_7x7, plane=plane)

        assert (fit.plane.galaxies[0].hyper_galaxy_image == lens_hyper_image).all()
        assert (fit_likelihood == fit.log_likelihood).all()
