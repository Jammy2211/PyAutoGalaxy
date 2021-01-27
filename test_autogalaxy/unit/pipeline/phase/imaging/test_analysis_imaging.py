from os import path

import autofit as af
import autogalaxy as ag
import pytest
from astropy import cosmology as cosmo
from autogalaxy.fit.fit import FitImaging
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestFit:
    def test__fit_using_imaging(self, imaging_7x7, mask_7x7, samples_with_result):

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
                source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
            ),
            search=mock.MockSearch(samples=samples_with_result, name="test_phase"),
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        assert isinstance(result.instance.galaxies[0], ag.Galaxy)
        assert isinstance(result.instance.galaxies[0], ag.Galaxy)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, imaging_7x7, mask_7x7
    ):
        galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=0.1))

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(galaxy=galaxy),
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(sub_size=1)
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        masked_imaging = ag.MaskedImaging(
            imaging=imaging_7x7,
            mask=mask_7x7,
            settings=ag.SettingsMaskedImaging(sub_size=1),
        )
        plane = analysis.plane_for_instance(instance=instance)

        fit = ag.FitImaging(masked_imaging=masked_imaging, plane=plane)

        assert fit.log_likelihood == fit_figure_of_merit

    def test__figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, imaging_7x7, mask_7x7
    ):
        hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        galalxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=0.1))

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(galaxy=galalxy),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(sub_size=4)
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        assert analysis.masked_imaging.mask.sub_size == 4

        masked_imaging = ag.MaskedImaging(
            imaging=imaging_7x7,
            mask=mask_7x7,
            settings=ag.SettingsMaskedImaging(sub_size=4),
        )
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
        galaxies.galaxy = ag.Galaxy(
            redshift=0.5,
            light=ag.lp.EllipticalSersic(intensity=1.0),
            mass=ag.mp.SphericalIsothermal,
        )
        galaxies.source = ag.Galaxy(redshift=1.0, light=ag.lp.EllipticalSersic())

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        galaxy_hyper_image = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=0.1)
        galaxy_hyper_image[4] = 10.0
        hyper_model_image = ag.Array2D.full(
            fill_value=0.5, shape_native=(3, 3), pixel_scales=0.1
        )

        hyper_galaxy_image_path_dict = {("galaxies", "galaxy"): galaxy_hyper_image}

        results = mock.MockResults(
            use_as_hyper_dataset=True,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=hyper_model_image,
        )

        analysis = ag.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            settings=ag.SettingsPhaseImaging(),
            results=results,
            cosmology=cosmo.Planck15,
        )

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

        fit = FitImaging(masked_imaging=masked_imaging_7x7, plane=plane)

        assert (fit.plane.galaxies[0].hyper_galaxy_image == galaxy_hyper_image).all()
        assert fit_likelihood == fit.log_likelihood
