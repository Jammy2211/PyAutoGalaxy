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
