import pytest

import autofit as af
import autogalaxy as ag
from autogalaxy.pipeline.phase.dataset import PhaseDataset
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)


class TestModel:
    def test__set_instances(self, phase_dataset_7x7):
        phase_dataset_7x7.galaxies = [ag.Galaxy(redshift=0.5)]
        assert phase_dataset_7x7.model.galaxies == [ag.Galaxy(redshift=0.5)]

    def test__set_models(self, phase_dataset_7x7):
        phase_dataset_7x7.galaxies = [ag.GalaxyModel(redshift=0.5)]
        assert phase_dataset_7x7.model.galaxies == [ag.GalaxyModel(redshift=0.5)]

    def test__promise_attrbutes(self):
        phase = PhaseDataset(
            galaxies=dict(
                galaxy=ag.GalaxyModel(
                    redshift=0.5,
                    mass=ag.mp.EllipticalIsothermal,
                    shear=ag.mp.ExternalShear,
                ),
                source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
            ),
            settings=ag.SettingsPhaseImaging(),
            search=mock.MockSearch(name="test_phase"),
        )

        print(hasattr(af.last.result.instance.galaxies.light, "mas2s"))

    def test__duplication(self):
        phase_dataset_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5), source=ag.GalaxyModel(redshift=1.0)
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        ag.PhaseImaging(search=mock.MockSearch(name="test_phase"))

        assert phase_dataset_7x7.galaxies is not None

    def test__phase_can_receive_list_of_galaxy_models(self):

        phase_dataset_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(
                    sersic=ag.lp.EllipticalSersic,
                    sis=ag.mp.SphericalIsothermal,
                    redshift=ag.Redshift,
                ),
                galaxy1=ag.GalaxyModel(
                    sis=ag.mp.SphericalIsothermal, redshift=ag.Redshift
                ),
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        for item in phase_dataset_7x7.model.path_priors_tuples:
            print(item)

        sersic = phase_dataset_7x7.model.galaxies[0].sersic
        sis = phase_dataset_7x7.model.galaxies[0].sis
        galaxy_1_sis = phase_dataset_7x7.model.galaxies[1].sis

        arguments = {
            sersic.centre[0]: 0.2,
            sersic.centre[1]: 0.2,
            sersic.elliptical_comps[0]: 0.0,
            sersic.elliptical_comps[1]: 0.1,
            sersic.effective_radius: 0.2,
            sersic.sersic_index: 0.6,
            sersic.intensity: 0.6,
            sis.centre[0]: 0.1,
            sis.centre[1]: 0.2,
            sis.einstein_radius: 0.3,
            phase_dataset_7x7.model.galaxies[0].redshift.priors[0]: 0.4,
            galaxy_1_sis.centre[0]: 0.6,
            galaxy_1_sis.centre[1]: 0.5,
            galaxy_1_sis.einstein_radius: 0.7,
            phase_dataset_7x7.model.galaxies[1].redshift.priors[0]: 0.8,
        }

        instance = phase_dataset_7x7.model.instance_for_arguments(arguments=arguments)

        assert instance.galaxies[0].sersic.centre[0] == 0.2
        assert instance.galaxies[0].sis.centre[0] == 0.1
        assert instance.galaxies[0].sis.centre[1] == 0.2
        assert instance.galaxies[0].sis.einstein_radius == 0.3
        assert instance.galaxies[0].redshift == 0.4
        assert instance.galaxies[1].sis.centre[0] == 0.6
        assert instance.galaxies[1].sis.centre[1] == 0.5
        assert instance.galaxies[1].sis.einstein_radius == 0.7
        assert instance.galaxies[1].redshift == 0.8

        class LensPlanePhase2(ag.PhaseImaging):
            # noinspection PyUnusedLocal
            def pass_models(self, results):
                self.galaxies[0].sis.einstein_radius = 10.0

        phase_dataset_7x7 = LensPlanePhase2(
            galaxies=dict(
                galaxy=ag.GalaxyModel(
                    sersic=ag.lp.EllipticalSersic,
                    sis=ag.mp.SphericalIsothermal,
                    redshift=ag.Redshift,
                ),
                galaxy1=ag.GalaxyModel(
                    sis=ag.mp.SphericalIsothermal, redshift=ag.Redshift
                ),
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        # noinspection PyTypeChecker
        phase_dataset_7x7.pass_models(None)

        sersic = phase_dataset_7x7.model.galaxies[0].sersic
        sis = phase_dataset_7x7.model.galaxies[0].sis
        galaxy_1_sis = phase_dataset_7x7.model.galaxies[1].sis

        arguments = {
            sersic.centre[0]: 0.01,
            sersic.centre[1]: 0.2,
            sersic.elliptical_comps[0]: 0.0,
            sersic.elliptical_comps[1]: 0.1,
            sersic.effective_radius: 0.2,
            sersic.sersic_index: 0.6,
            sersic.intensity: 0.6,
            sis.centre[0]: 0.1,
            sis.centre[1]: 0.2,
            phase_dataset_7x7.model.galaxies[0].redshift.priors[0]: 0.4,
            galaxy_1_sis.centre[0]: 0.6,
            galaxy_1_sis.centre[1]: 0.5,
            galaxy_1_sis.einstein_radius: 0.7,
            phase_dataset_7x7.model.galaxies[1].redshift.priors[0]: 0.8,
        }

        instance = phase_dataset_7x7.model.instance_for_arguments(arguments)

        assert instance.galaxies[0].sersic.centre[0] == 0.01
        assert instance.galaxies[0].sis.centre[0] == 0.1
        assert instance.galaxies[0].sis.centre[1] == 0.2
        assert instance.galaxies[0].sis.einstein_radius == 10.0
        assert instance.galaxies[0].redshift == 0.4
        assert instance.galaxies[1].sis.centre[0] == 0.6
        assert instance.galaxies[1].sis.centre[1] == 0.5
        assert instance.galaxies[1].sis.einstein_radius == 0.7
        assert instance.galaxies[1].redshift == 0.8

    def test__pixelization_property_extracts_pixelization(self, imaging_7x7, mask_7x7):

        source_galaxy = ag.Galaxy(redshift=0.5)

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
        )

        assert phase_imaging_7x7.pixelization is None
        assert phase_imaging_7x7.has_pixelization is False
        assert phase_imaging_7x7.pixelization_is_model == False

        source_galaxy = ag.Galaxy(
            redshift=0.5,
            pixelization=ag.pix.Rectangular(),
            regularization=ag.reg.Constant(),
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
        )

        assert isinstance(phase_imaging_7x7.pixelization, ag.pix.Rectangular)
        assert phase_imaging_7x7.has_pixelization is True
        assert phase_imaging_7x7.pixelization_is_model == False

        source_galaxy = ag.GalaxyModel(
            redshift=0.5,
            pixelization=ag.pix.Rectangular,
            regularization=ag.reg.Constant,
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
        )

        assert type(phase_imaging_7x7.pixelization) == type(ag.pix.Rectangular)
        assert phase_imaging_7x7.has_pixelization is True
        assert phase_imaging_7x7.pixelization_is_model == True

        pixelization = af.PriorModel(ag.pix.VoronoiBrightnessImage)
        pixelization.pixels = 100

        source_galaxy = ag.GalaxyModel(
            redshift=0.5, pixelization=pixelization, regularization=ag.reg.Constant
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
        )

        assert type(phase_imaging_7x7.pixelization) == type(ag.pix.Rectangular)
        assert phase_imaging_7x7.has_pixelization is True
        assert phase_imaging_7x7.pixelization_is_model == True

    def test__check_if_phase_uses_cluster_inversion(self):
        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5), source=ag.GalaxyModel(redshift=1.0)
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is False

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(
                    redshift=0.5,
                    pixelization=ag.pix.Rectangular,
                    regularization=ag.reg.Constant,
                ),
                source=ag.GalaxyModel(redshift=1.0),
            ),
            search=mock.MockSearch(name="test_phase"),
        )
        assert phase_imaging_7x7.uses_cluster_inversion is False

        source = ag.GalaxyModel(
            redshift=1.0,
            pixelization=ag.pix.VoronoiBrightnessImage,
            regularization=ag.reg.Constant,
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(galaxy=ag.GalaxyModel(redshift=0.5), source=source),
            search=mock.MockSearch(name="test_phase"),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is True

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5), source=ag.GalaxyModel(redshift=1.0)
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is False

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(
                    redshift=0.5,
                    pixelization=ag.pix.Rectangular,
                    regularization=ag.reg.Constant,
                ),
                source=ag.GalaxyModel(redshift=1.0),
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is False

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5),
                source=ag.GalaxyModel(
                    redshift=1.0,
                    pixelization=ag.pix.VoronoiBrightnessImage,
                    regularization=ag.reg.Constant,
                ),
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is True

        pixelization = af.PriorModel(ag.pix.VoronoiBrightnessImage)
        pixelization.pixels = 100

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5),
                source=ag.GalaxyModel(
                    redshift=1.0,
                    pixelization=pixelization,
                    regularization=ag.reg.Constant,
                ),
            ),
            search=mock.MockSearch(name="test_phase"),
        )

        assert phase_imaging_7x7.uses_cluster_inversion is True


class TestSetup:

    # noinspection PyTypeChecker
    def test_assertion_failure(self, imaging_7x7, mask_7x7):
        phase_dataset_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.Galaxy(light=ag.lp.EllipticalLightProfile, redshift=1)
            ),
            settings=ag.SettingsPhaseImaging(),
            search=mock.MockSearch(name="name"),
        )

        result = phase_dataset_7x7.run(dataset=imaging_7x7, mask=mask_7x7, results=None)
        assert result is not None

        phase_dataset_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.Galaxy(light=ag.lp.EllipticalLightProfile, redshift=1)
            ),
            settings=ag.SettingsPhaseImaging(),
            search=mock.MockSearch(name="name"),
        )
        result = phase_dataset_7x7.run(dataset=imaging_7x7, mask=mask_7x7, results=None)
        assert result is not None
