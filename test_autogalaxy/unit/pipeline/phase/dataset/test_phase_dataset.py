from os import path

import pytest

import autogalaxy as ag
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestPhase:
    def test__extend_with_hyper_phase(self):

        phase_with_hyper_sky = ag.PhaseImaging(search=mock.MockSearch())

        setup_hyper = ag.SetupHyper(
            hyper_image_sky=ag.hyper_data.HyperImageSky,
            hyper_search_with_inversion=mock.MockSearch("test_phase"),
        )

        phase_extended = phase_with_hyper_sky.extend_with_hyper_phase(
            setup_hyper=setup_hyper
        )

        assert isinstance(phase_extended, ag.HyperPhase)
        assert phase_extended.hyper_image_sky is ag.hyper_data.HyperImageSky

        phase_with_hyper_sky = ag.PhaseImaging(search=mock.MockSearch())

        phase_extended = phase_with_hyper_sky.extend_with_hyper_phase(
            setup_hyper=ag.SetupHyper(
                hyper_background_noise=ag.hyper_data.HyperBackgroundNoise,
                hyper_search_with_inversion=mock.MockSearch("test_phase"),
            )
        )

        assert isinstance(phase_extended, ag.HyperPhase)

        phase_with_pixelization = ag.PhaseImaging(
            galaxies=dict(
                source=ag.GalaxyModel(
                    redshift=0.5,
                    pixelization=ag.pix.Rectangular,
                    regularization=ag.reg.Constant,
                )
            ),
            search=mock.MockSearch(),
        )

        phase_extended = phase_with_pixelization.extend_with_hyper_phase(
            setup_hyper=ag.SetupHyper(
                hyper_search_with_inversion=mock.MockSearch("test_phase")
            )
        )

        assert isinstance(phase_extended, ag.HyperPhase)

    def test__extend_with_hyper_phase__does_not_extend_if_no_hyper_compoennts_to_optimize(
        self
    ):

        phase_no_pixelization = ag.PhaseImaging(search=mock.MockSearch("test_phase"))

        phase_extended = phase_no_pixelization.extend_with_hyper_phase(
            setup_hyper=ag.SetupHyper()
        )

        assert phase_extended == phase_no_pixelization

        phase_no_pixelization = ag.PhaseImaging(search=mock.MockSearch("test_phase"))

        phase_extended = phase_no_pixelization.extend_with_hyper_phase(
            setup_hyper=ag.SetupHyper(hyper_image_sky=ag.hyper_data.HyperImageSky),
            include_hyper_image_sky=False,
        )

        assert phase_extended == phase_no_pixelization


class TestMakeAnalysis:
    def test__mask_input_uses_mask(self, phase_imaging_7x7, imaging_7x7):
        # If an input mask is supplied we use mask input.

        mask_input = ag.Mask2D.circular(
            shape_native=imaging_7x7.shape_native,
            pixel_scales=1.0,
            sub_size=1,
            radius=1.5,
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_input, results=mock.MockResults()
        )

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.pixel_scales == mask_input.pixel_scales

    def test__mask_changes_sub_size_depending_on_phase_attribute(
        self, phase_imaging_7x7, imaging_7x7
    ):
        # If an input mask is supplied we use mask input.

        mask_input = ag.Mask2D.circular(
            shape_native=imaging_7x7.shape_native,
            pixel_scales=1,
            sub_size=1,
            radius=1.5,
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(sub_size=1)
            ),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_input, results=mock.MockResults()
        )

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.sub_size == 1
        assert analysis.masked_imaging.mask.pixel_scales == mask_input.pixel_scales

        phase_imaging_7x7 = ag.PhaseImaging(
            search=mock.MockSearch("test_phase"),
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(sub_size=2)
            ),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_input, results=mock.MockResults()
        )

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.sub_size == 2
        assert analysis.masked_imaging.mask.pixel_scales == mask_input.pixel_scales


class TestPhasePickle:

    # noinspection PyTypeChecker
    def test_assertion_failure(self, imaging_7x7, mask_7x7):
        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.Galaxy(light=ag.lp.EllipticalLightProfile, redshift=1)
            ),
            search=mock.MockSearch("name"),
        )

        result = phase_imaging_7x7.run(dataset=imaging_7x7, mask=mask_7x7, results=None)
        assert result is not None

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.Galaxy(light=ag.lp.EllipticalLightProfile, redshift=1)
            ),
            search=mock.MockSearch("name"),
        )

        result = phase_imaging_7x7.run(dataset=imaging_7x7, mask=mask_7x7, results=None)
        assert result is not None
