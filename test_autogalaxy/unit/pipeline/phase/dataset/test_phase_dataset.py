from os import path

import autogalaxy as ag
import pytest
from astropy import cosmology as cosmo
from autogalaxy import exc
from test_autogalaxy.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestPhase:
    def test__extend_with_hyper_and_pixelizations(self):

        phase_no_pixelization = ag.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO, phase_name="test_phase"
        )

        phase_extended = phase_no_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=False
        )
        assert phase_extended == phase_no_pixelization

        # This phase does not have a pixelization, so even though inversion=True it will not be extended

        phase_extended = phase_no_pixelization.extend_with_multiple_hyper_phases(
            inversion=True
        )
        assert phase_extended == phase_no_pixelization

        phase_with_pixelization = ag.PhaseImaging(
            galaxies=dict(
                source=ag.GalaxyModel(
                    redshift=0.5,
                    pixelization=ag.pix.Rectangular,
                    regularization=ag.reg.Constant,
                )
            ),
            non_linear_class=mock_pipeline.MockNLO,
            phase_name="test_phase",
        )

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == ag.InversionPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=False
        )
        assert type(phase_extended.hyper_phases[0]) == ag.HyperGalaxyPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=False, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == ag.InversionPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True
        )
        assert type(phase_extended.hyper_phases[0]) == ag.InversionPhase
        assert type(phase_extended.hyper_phases[1]) == ag.HyperGalaxyPhase

        phase_extended = phase_with_pixelization.extend_with_multiple_hyper_phases(
            hyper_galaxy=True, inversion=True, hyper_galaxy_phase_first=True
        )
        assert type(phase_extended.hyper_phases[0]) == ag.HyperGalaxyPhase
        assert type(phase_extended.hyper_phases[1]) == ag.InversionPhase


class TestMakeAnalysis:
    def test__mask_input_uses_mask(self, phase_imaging_7x7, imaging_7x7):
        # If an input mask is supplied we use mask input.

        mask_input = ag.Mask.circular(
            shape_2d=imaging_7x7.shape_2d, pixel_scales=1.0, sub_size=1, radius=1.5
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_input, results=mock_pipeline.MockResults()
        )

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.pixel_scales == mask_input.pixel_scales

    def test__mask_changes_sub_size_depending_on_phase_attribute(
        self, phase_imaging_7x7, imaging_7x7
    ):
        # If an input mask is supplied we use mask input.

        mask_input = ag.Mask.circular(
            shape_2d=imaging_7x7.shape_2d, pixel_scales=1, sub_size=1, radius=1.5
        )

        phase_imaging_7x7.meta_dataset.sub_size = 1
        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_input, results=mock_pipeline.MockResults()
        )

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.sub_size == 1
        assert analysis.masked_imaging.mask.pixel_scales == mask_input.pixel_scales

        phase_imaging_7x7.meta_dataset.sub_size = 2
        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_input, results=mock_pipeline.MockResults()
        )

        assert (analysis.masked_imaging.mask == mask_input).all()
        assert analysis.masked_imaging.mask.sub_size == 2
        assert analysis.masked_imaging.mask.pixel_scales == mask_input.pixel_scales

    def test__inversion_resolution_error_raised_if_above_inversion_pixel_limit(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                source=ag.Galaxy(
                    redshift=0.5,
                    pixelization=ag.pix.Rectangular(shape=(3, 3)),
                    regularization=ag.reg.Constant(),
                )
            ),
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        plane = analysis.plane_for_instance(instance=instance)

        analysis.masked_dataset.check_inversion_pixels_are_below_limit_via_plane(
            plane=plane
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                source=ag.Galaxy(
                    redshift=0.5,
                    pixelization=ag.pix.Rectangular(shape=(4, 4)),
                    regularization=ag.reg.Constant(),
                )
            ),
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        plane = analysis.plane_for_instance(instance=instance)

        with pytest.raises(exc.PixelizationException):
            analysis.masked_dataset.check_inversion_pixels_are_below_limit_via_plane(
                plane=plane
            )
            analysis.log_likelihood_function(instance=instance)

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                source=ag.Galaxy(
                    redshift=0.5,
                    pixelization=ag.pix.Rectangular(shape=(3, 3)),
                    regularization=ag.reg.Constant(),
                )
            ),
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        plane = analysis.plane_for_instance(instance=instance)

        analysis.masked_dataset.check_inversion_pixels_are_below_limit_via_plane(
            plane=plane
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                source=ag.Galaxy(
                    redshift=0.5,
                    pixelization=ag.pix.Rectangular(shape=(4, 4)),
                    regularization=ag.reg.Constant(),
                )
            ),
            inversion_pixel_limit=10,
            cosmology=cosmo.FLRW,
            phase_name="test_phase",
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )
        instance = phase_imaging_7x7.model.instance_from_unit_vector([])
        plane = analysis.plane_for_instance(instance=instance)

        with pytest.raises(exc.PixelizationException):
            analysis.masked_dataset.check_inversion_pixels_are_below_limit_via_plane(
                plane=plane
            )
            analysis.log_likelihood_function(instance=instance)


class TestPhasePickle:

    # noinspection PyTypeChecker
    def test_assertion_failure(self, imaging_7x7, mask_7x7):
        def make_analysis(*args, **kwargs):
            return mock_pipeline.GalaxiesMockAnalysis(1, 1)

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                galaxy=ag.Galaxy(light=ag.lp.EllipticalLightProfile, redshift=1)
            ),
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(dataset=imaging_7x7, mask=mask_7x7, results=None)
        assert result is not None

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                galaxy=ag.Galaxy(light=ag.lp.EllipticalLightProfile, redshift=1)
            ),
        )

        phase_imaging_7x7.make_analysis = make_analysis
        result = phase_imaging_7x7.run(dataset=imaging_7x7, mask=mask_7x7, results=None)
        assert result is not None

        class CustomPhase(ag.PhaseImaging):
            def customize_priors(self, results):
                self.galaxies.light.light = ag.lp.EllipticalLightProfile()

        phase_imaging_7x7 = CustomPhase(
            phase_name="phase_name",
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                galaxy=ag.Galaxy(light=ag.lp.EllipticalLightProfile, redshift=1)
            ),
        )
        phase_imaging_7x7.make_analysis = make_analysis

        # with pytest.raises(af.exc.PipelineException):
        #     phase_imaging_7x7.run(data=imaging_7x7, results=None, mask=None, positions=None)
