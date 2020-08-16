from os import path
import autofit as af
import autogalaxy as ag
import pytest
from test_autogalaxy import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestSetup:
    def test__pixelization_property_extracts_pixelization(self, imaging_7x7, mask_7x7):
        source_galaxy = ag.Galaxy(redshift=0.5)

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase", galaxies=[source_galaxy], search=mock.MockSearch()
        )

        assert phase_imaging_7x7.meta_dataset.pixelization is None
        assert phase_imaging_7x7.meta_dataset.has_pixelization is False
        assert phase_imaging_7x7.meta_dataset.pixelization_is_model == False

        source_galaxy = ag.Galaxy(
            redshift=0.5,
            pixelization=ag.pix.Rectangular(),
            regularization=ag.reg.Constant(),
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase", galaxies=[source_galaxy], search=mock.MockSearch()
        )

        assert isinstance(
            phase_imaging_7x7.meta_dataset.pixelization, ag.pix.Rectangular
        )
        assert phase_imaging_7x7.meta_dataset.has_pixelization is True
        assert phase_imaging_7x7.meta_dataset.pixelization_is_model == False

        source_galaxy = ag.GalaxyModel(
            redshift=0.5,
            pixelization=ag.pix.Rectangular,
            regularization=ag.reg.Constant,
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase", galaxies=[source_galaxy], search=mock.MockSearch()
        )

        assert type(phase_imaging_7x7.meta_dataset.pixelization) == type(
            ag.pix.Rectangular
        )
        assert phase_imaging_7x7.meta_dataset.has_pixelization is True
        assert phase_imaging_7x7.meta_dataset.pixelization_is_model == True

        pixelization = af.PriorModel(ag.pix.VoronoiBrightnessImage)
        pixelization.pixels = 100

        source_galaxy = ag.GalaxyModel(
            redshift=0.5, pixelization=pixelization, regularization=ag.reg.Constant
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase", galaxies=[source_galaxy], search=mock.MockSearch()
        )

        assert type(phase_imaging_7x7.meta_dataset.pixelization) == type(
            ag.pix.Rectangular
        )
        assert phase_imaging_7x7.meta_dataset.has_pixelization is True
        assert phase_imaging_7x7.meta_dataset.pixelization_is_model == True

    def test__check_if_phase_uses_cluster_inversion(self):
        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5), source=ag.GalaxyModel(redshift=1.0)
            ),
            search=mock.MockSearch(),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is False

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                galaxy=ag.GalaxyModel(
                    redshift=0.5,
                    pixelization=ag.pix.Rectangular,
                    regularization=ag.reg.Constant,
                ),
                source=ag.GalaxyModel(redshift=1.0),
            ),
            search=mock.MockSearch(),
        )
        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is False

        source = ag.GalaxyModel(
            redshift=1.0,
            pixelization=ag.pix.VoronoiBrightnessImage,
            regularization=ag.reg.Constant,
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(galaxy=ag.GalaxyModel(redshift=0.5), source=source),
            search=mock.MockSearch(),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is True

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5), source=ag.GalaxyModel(redshift=1.0)
            ),
            search=mock.MockSearch(),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is False

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                galaxy=ag.GalaxyModel(
                    redshift=0.5,
                    pixelization=ag.pix.Rectangular,
                    regularization=ag.reg.Constant,
                ),
                source=ag.GalaxyModel(redshift=1.0),
            ),
            search=mock.MockSearch(),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is False

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5),
                source=ag.GalaxyModel(
                    redshift=1.0,
                    pixelization=ag.pix.VoronoiBrightnessImage,
                    regularization=ag.reg.Constant,
                ),
            ),
            search=mock.MockSearch(),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is True

        pixelization = af.PriorModel(ag.pix.VoronoiBrightnessImage)
        pixelization.pixels = 100

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5),
                source=ag.GalaxyModel(
                    redshift=1.0,
                    pixelization=pixelization,
                    regularization=ag.reg.Constant,
                ),
            ),
            search=mock.MockSearch(),
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is True
