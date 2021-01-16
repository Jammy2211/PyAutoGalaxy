from os import path
import autofit as af
import autofit.non_linear.paths
import autogalaxy as ag
from autogalaxy.hyper import hyper_data as hd
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
        phase = ag.HyperPhase(
            phase=MockPhase(),
            hyper_search=mock.MockSearch(),
            hyper_image_sky=None,
            hyper_background_noise=None,
        )

        instance = af.ModelInstance()

        mapper = phase.make_model(instance=instance)

        assert mapper.hyper_image_sky is None
        assert mapper.hyper_background_noise is None

        # noinspection PyTypeChecker
        phase = ag.HyperPhase(
            phase=MockPhase(),
            hyper_search=mock.MockSearch(),
            hyper_image_sky=hd.HyperImageSky,
            hyper_background_noise=hd.HyperBackgroundNoise,
        )

        instance = af.ModelInstance()

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
        phase = ag.HyperPhase(
            phase=MockPhase(),
            hyper_search=mock.MockSearch(),
            model_classes=(ag.pix.Pixelization, ag.reg.Regularization),
        )

        mapper = mapper.copy_with_fixed_priors(instance, phase.model_classes)

        assert mapper.prior_count == 3
        assert mapper.galaxy.redshift == 1.0
        assert mapper.source_galaxy.light.elliptical_comps[0] == 0.0

    def test__hyper_phase_result(self):

        phase = ag.HyperPhase(
            phase=MockPhase(),
            hyper_search=mock.MockSearch(),
            model_classes=(ag.pix.Pixelization, ag.reg.Regularization),
        )

        result = phase.run(dataset=None, mask=None, results=af.ResultsCollection())

        assert hasattr(result, "hyper")
        assert isinstance(result.hyper, mock.MockResult)

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

        phase_extended = phase.extend_with_hyper_phase(
            setup_hyper=ag.SetupHyper(
                hyper_galaxies=False,
                hyper_search_with_inversion=af.DynestyStatic(n_live_points=1),
            )
        )

        hyper_phase = phase_extended.make_hyper_phase()
        hyper_phase.modify_search_paths()

        assert (
            path.join(
                "test_phase",
                "hyper__settings__imaging[grid_sub_2_inv_sub_2__bin_2]__pix[use_border]__inv[mat]",
                "dynesty_static[nlive_1",
            )
            in hyper_phase.paths.output_path
        )
