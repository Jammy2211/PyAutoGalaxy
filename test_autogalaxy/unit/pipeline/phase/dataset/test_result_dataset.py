from os import path

import autogalaxy as ag
import numpy as np
import pytest
from test_autolens.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestResult:
    def test__results_of_phase_are_available_as_properties(self, imaging_7x7, mask_7x7):

        phase_imaging_7x7 = ag.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=[
                ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0))
            ],
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert isinstance(result, ag.AbstractPhase.Result)

    def test__results_of_phase_include_mask__available_as_property(
        self, imaging_7x7, mask_7x7
    ):

        phase_imaging_7x7 = ag.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=[
                ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0))
            ],
            sub_size=2,
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert (result.mask == mask_7x7).all()

    def test__results_of_phase_include_pixelization__available_as_property(
        self, imaging_7x7, mask_7x7
    ):

        phase_imaging_7x7 = ag.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=ag.Galaxy(
                    redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0)
                ),
                source=ag.Galaxy(
                    redshift=1.0,
                    pixelization=ag.pix.VoronoiMagnification(shape=(2, 3)),
                    regularization=ag.reg.Constant(),
                ),
            ),
            inversion_pixel_limit=6,
            phase_name="test_phase_2",
        )

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert isinstance(result.pixelization, ag.pix.VoronoiMagnification)
        assert result.pixelization.shape == (2, 3)

        phase_imaging_7x7 = ag.PhaseImaging(
            non_linear_class=mock_pipeline.MockNLO,
            galaxies=dict(
                lens=ag.Galaxy(
                    redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0)
                ),
                source=ag.Galaxy(
                    redshift=1.0,
                    pixelization=ag.pix.VoronoiBrightnessImage(pixels=6),
                    regularization=ag.reg.Constant(),
                ),
            ),
            inversion_pixel_limit=6,
            phase_name="test_phase_2",
        )

        phase_imaging_7x7.galaxies.source.hyper_galaxy_image = np.ones(9)

        result = phase_imaging_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert isinstance(result.pixelization, ag.pix.VoronoiBrightnessImage)
        assert result.pixelization.pixels == 6
