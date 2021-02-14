from os import path

import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestResult:
    def test__results_of_phase_are_available_as_properties(self, imaging_7x7, mask_7x7):
        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.Galaxy(
                    redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0)
                )
            ),
            search=mock.MockSearch("test_phase_2"),
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(result, ag.AbstractPhase.Result)

    def test__results_of_phase_include_mask__available_as_property(
        self, imaging_7x7, mask_7x7, samples_with_result
    ):
        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.Galaxy(
                    redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0)
                )
            ),
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(sub_size=2)
            ),
            search=mock.MockSearch("test_phase_2", samples=samples_with_result),
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert (result.mask == mask_7x7).all()

    def test__results_of_phase_include_pixelization__available_as_property(
        self, imaging_7x7, mask_7x7
    ):
        source = ag.Galaxy(
            redshift=1.0,
            pixelization=ag.pix.VoronoiMagnification(shape=(2, 3)),
            regularization=ag.reg.Constant(),
        )

        max_log_likelihood_plane = ag.Plane(galaxies=[source])

        samples = mock.MockSamples(max_log_likelihood_instance=max_log_likelihood_plane)

        phase_imaging_7x7 = ag.PhaseImaging(
            settings=ag.SettingsPhaseImaging(),
            search=mock.MockSearch("test_phase_2", samples=samples),
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(result.pixelization, ag.pix.VoronoiMagnification)
        assert result.pixelization.shape == (2, 3)

        source = ag.Galaxy(
            redshift=1.0,
            pixelization=ag.pix.VoronoiBrightnessImage(pixels=6),
            regularization=ag.reg.Constant(),
        )

        source.hyper_galaxy_image = np.ones(9)

        max_log_likelihood_plane = ag.Plane(galaxies=[source])

        samples = mock.MockSamples(max_log_likelihood_instance=max_log_likelihood_plane)

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(source=source),
            settings=ag.SettingsPhaseImaging(),
            search=mock.MockSearch("test_phase_2", samples=samples),
        )

        phase_imaging_7x7.galaxies.source.hyper_galaxy_image = np.ones(9)

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(result.pixelization, ag.pix.VoronoiBrightnessImage)
        assert result.pixelization.pixels == 6
