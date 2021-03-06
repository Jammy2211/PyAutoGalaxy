import pytest

import autofit as af
import autogalaxy as ag
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)


class TestModel:
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
        analysis_dataset_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.Galaxy(light=ag.lp.EllipticalLightProfile, redshift=1)
            ),
            settings=ag.SettingsPhaseImaging(),
            search=mock.MockSearch(name="name"),
        )

        result = analysis_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=None
        )
        assert result is not None

        analysis_dataset_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.Galaxy(light=ag.lp.EllipticalLightProfile, redshift=1)
            ),
            settings=ag.SettingsPhaseImaging(),
            search=mock.MockSearch(name="name"),
        )
        result = analysis_dataset_7x7.run(
            dataset=imaging_7x7, mask=mask_7x7, results=None
        )
        assert result is not None
