from os import path
import autofit as af
import autofit.non_linear.paths
import autogalaxy as ag
from autogalaxy.hyper import hyper_data as hd
import pytest
from autogalaxy.fit.fit import FitImaging
from autogalaxy.mock import mock


class MockPhase:
    def __init__(self):
        self.name = "name"
        self.paths = autofit.non_linear.paths.Paths(
            name=self.name, path_prefix="phase_path", tag=""
        )
        self.search = mock.MockSearch(paths=self.paths)
        self.model = af.ModelMapper()

    def save_dataset(self, dataset):
        pass

    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    def run(self, *args, **kwargs):
        return mock.MockResult()


class TestModelFixing:
    def test__defaults_both(self):
        # noinspection PyTypeChecker
        phase = ag.ModelFixingHyperPhase(
            phase=MockPhase(),
            hyper_name="test",
            hyper_search=mock.MockSearch(),
            model_classes=(hd.HyperImageSky, hd.HyperBackgroundNoise),
        )

        instance = af.ModelInstance()
        instance.hyper_image_sky = ag.hyper_data.HyperImageSky()
        instance.hyper_background_noise = ag.hyper_data.HyperBackgroundNoise()

        mapper = phase.make_model(instance=instance)

        assert isinstance(mapper.hyper_image_sky, af.PriorModel)
        assert isinstance(mapper.hyper_background_noise, af.PriorModel)

        assert mapper.hyper_image_sky.cls == ag.hyper_data.HyperImageSky
        assert mapper.hyper_background_noise.cls == ag.hyper_data.HyperBackgroundNoise

    def test__make_pixelization_model(self):
        instance = af.ModelInstance()
        mapper = af.ModelMapper()

        mapper.galaxy = ag.GalaxyModel(
            redshift=ag.Redshift,
            pixelization=ag.pix.Rectangular,
            regularization=ag.reg.Constant,
        )
        mapper.source_galaxy = ag.GalaxyModel(
            redshift=ag.Redshift, light=ag.lp.EllipticalLightProfile
        )

        assert mapper.prior_count == 10

        instance.galaxy = ag.Galaxy(
            pixelization=ag.pix.Rectangular(),
            regularization=ag.reg.Constant(),
            redshift=1.0,
        )
        instance.source_galaxy = ag.Galaxy(
            redshift=1.0, light=ag.lp.EllipticalLightProfile()
        )

        # noinspection PyTypeChecker
        phase = ag.ModelFixingHyperPhase(
            phase=MockPhase(),
            hyper_name="mock_phase",
            hyper_search=mock.MockSearch(),
            model_classes=(ag.pix.Pixelization, ag.reg.Regularization),
        )

        mapper = mapper.copy_with_fixed_priors(instance, phase.model_classes)

        assert mapper.prior_count == 3
        assert mapper.galaxy.redshift == 1.0
        assert mapper.source_galaxy.light.elliptical_comps[0] == 0.0


@pytest.fixture(name="hyper_combined")
def make_combined():
    normal_phase = MockPhase()

    # noinspection PyUnusedLocal
    def run_hyper(*args, **kwargs):
        return mock.MockResult()

    hyper_galaxy_phase = ag.HyperGalaxyPhase(
        phase=normal_phase,
        hyper_search=mock.MockSearch(),
        include_sky_background=False,
        include_noise_background=False,
    )
    inversion_phase = ag.InversionPhase(
        phase=normal_phase,
        hyper_search=mock.MockSearch(),
        model_classes=(ag.pix.Pixelization, ag.reg.Regularization),
    )

    # noinspection PyTypeChecker
    hyper_combined = ag.CombinedHyperPhase(
        phase=normal_phase,
        hyper_phases=(hyper_galaxy_phase, inversion_phase),
        hyper_search=mock.MockSearch(),
    )

    for phase in hyper_combined.hyper_phases:
        phase.run_hyper = run_hyper

    return hyper_combined


class TestHyperAPI:
    def test_combined_result(self, hyper_combined):
        result = hyper_combined.run(
            dataset=None, mask=None, results=af.ResultsCollection()
        )

        assert hasattr(result, "hyper_galaxy")
        assert isinstance(result.hyper_galaxy, mock.MockResult)

        assert hasattr(result, "inversion")
        assert isinstance(result.inversion, mock.MockResult)

        assert hasattr(result, "hyper_combined")
        assert isinstance(result.hyper_combined, mock.MockResult)

    def test_combine_models(self, hyper_combined):
        result = mock.MockResult()
        hyper_galaxy_result = mock.MockResult()
        inversion_result = mock.MockResult()

        hyper_galaxy_result.model = af.ModelMapper()
        inversion_result.model = af.ModelMapper()

        hyper_galaxy_result.model.hyper_galaxy = ag.HyperGalaxy
        hyper_galaxy_result.model.pixelization = ag.pix.Pixelization()
        inversion_result.model.pixelization = ag.pix.Pixelization
        inversion_result.model.hyper_galaxy = ag.HyperGalaxy()

        result.hyper_galaxy = hyper_galaxy_result
        result.inversion = inversion_result

        model = hyper_combined.combine_models(result)

        assert isinstance(model.hyper_galaxy, af.PriorModel)
        assert isinstance(model.pixelization, af.PriorModel)

        assert model.hyper_galaxy.cls == ag.HyperGalaxy
        assert model.pixelization.cls == ag.pix.Pixelization

    def test_instantiation(self, hyper_combined):
        assert len(hyper_combined.hyper_phases) == 2

        galaxy_phase = hyper_combined.hyper_phases[0]
        pixelization_phase = hyper_combined.hyper_phases[1]

        assert galaxy_phase.hyper_name == "hyper_galaxy"
        assert isinstance(galaxy_phase, ag.HyperGalaxyPhase)

        assert pixelization_phase.hyper_name == "inversion"
        assert isinstance(pixelization_phase, ag.InversionPhase)

    def test_hyper_result(self, imaging_7x7, mask_7x7):
        normal_phase = MockPhase()

        # noinspection PyTypeChecker
        phase = ag.HyperGalaxyPhase(
            phase=normal_phase,
            hyper_search=mock.MockSearch(),
            include_sky_background=False,
            include_noise_background=False,
        )

        # noinspection PyUnusedLocal
        def run_hyper(*args, **kwargs):
            return mock.MockResult()

        phase.run_hyper = run_hyper

        result = phase.run(dataset=imaging_7x7, results=af.ResultsCollection())

        assert hasattr(result, "hyper_galaxy")
        assert isinstance(result.hyper_galaxy, mock.MockResult)

    def test__paths(self):

        galaxy = ag.Galaxy(
            pixelization=ag.pix.Rectangular(),
            regularization=ag.reg.Constant(),
            redshift=1.0,
        )

        phase = ag.PhaseImaging(
            galaxies=dict(galaxy=galaxy),
            search=af.DynestyStatic(n_live_points=1, name="test_phase"),
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(bin_up_factor=2)
            ),
        )

        phase_extended = phase.extend_with_inversion_phase(
            hyper_search=af.DynestyStatic(n_live_points=1)
        )

        phase_extended.paths.path_prefix = "prefix"
        hyper_phase = phase_extended.make_hyper_phase()
        hyper_phase.modify_search_paths()

        assert (
            path.join(
                "output",
                "prefix",
                "test_phase",
                "inversion__settings__imaging[grid_sub_2_inv_sub_2__bin_2]__pix[no_border]__inv[mat]",
                "dynesty_static[nlive_1",
            )
            in hyper_phase.paths.output_path
        )

        phase_extended = phase.extend_with_multiple_hyper_phases(
            setup_hyper=ag.SetupHyper(
                hyper_galaxies=True,
                inversion_search=af.DynestyStatic(n_live_points=1),
                hyper_galaxies_search=af.DynestyStatic(n_live_points=2),
                hyper_combined_search=af.DynestyStatic(n_live_points=3),
            ),
            include_inversion=True,
        )

        inversion_phase = phase_extended.hyper_phases[0].make_hyper_phase()
        inversion_phase.modify_search_paths()

        assert (
            path.join(
                "test_phase",
                "inversion__settings__imaging[grid_sub_2_inv_sub_2__bin_2]__pix[no_border]__inv[mat]",
                "dynesty_static[nlive_1",
            )
            in inversion_phase.paths.output_path
        )

        hyper_galaxy_phase = phase_extended.hyper_phases[1].make_hyper_phase()
        hyper_galaxy_phase.modify_search_paths()

        assert (
            path.join(
                "test_phase",
                "hyper_galaxy__settings__imaging[grid_sub_2_inv_sub_2__bin_2]__pix[no_border]__inv[mat]",
                "dynesty_static[nlive_2",
            )
            in hyper_galaxy_phase.paths.output_path
        )

        hyper_combined_phase = phase_extended.make_hyper_phase()
        hyper_combined_phase.modify_search_paths()

        assert (
            path.join(
                "test_phase",
                "hyper_combined__settings__imaging[grid_sub_2_inv_sub_2__bin_2]__pix[no_border]__inv[mat]",
                "dynesty_static[nlive_3",
            )
            in hyper_combined_phase.paths.output_path
        )


class TestHyperGalaxyPhase:
    def test__likelihood_function_is_same_as_normal_phase_likelihood_function(
        self, imaging_7x7, mask_7x7
    ):

        hyper_image_sky = ag.hyper_data.HyperImageSky(sky_scale=1.0)
        hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=0.1))

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(galaxy=galaxy),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(sub_size=2)
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])

        assert analysis.masked_dataset.mask.sub_size == 2

        masked_imaging = ag.MaskedImaging(
            imaging=imaging_7x7,
            mask=mask_7x7,
            settings=ag.SettingsMaskedImaging(sub_size=2),
        )
        plane = analysis.plane_for_instance(instance=instance)
        fit = FitImaging(
            masked_imaging=masked_imaging,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        phase_imaging_7x7_hyper = phase_imaging_7x7.extend_with_multiple_hyper_phases(
            setup_hyper=ag.SetupHyper(
                hyper_galaxies=True, hyper_galaxies_search=mock.MockSearch()
            )
        )

        instance = phase_imaging_7x7_hyper.model.instance_from_unit_vector([])

        instance.hyper_galaxy = ag.HyperGalaxy(noise_factor=0.0)

        analysis = phase_imaging_7x7_hyper.hyper_phases[0].Analysis(
            masked_imaging=masked_imaging,
            hyper_model_image=fit.model_image,
            hyper_galaxy_image=fit.model_image,
        )

        fit_hyper = analysis.fit_for_hyper_galaxy(
            hyper_galaxy=ag.HyperGalaxy(noise_factor=0.0),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit_hyper.log_likelihood == fit.log_likelihood

        fit_hyper = analysis.fit_for_hyper_galaxy(
            hyper_galaxy=ag.HyperGalaxy(noise_factor=1.0),
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit_hyper.log_likelihood != fit.log_likelihood

        instance.hyper_galaxy = ag.HyperGalaxy(noise_factor=0.0)

        log_likelihood = analysis.log_likelihood_function(instance=instance)

        assert log_likelihood == fit.log_likelihood
