from os import path

import autogalaxy as ag
import pytest
from astropy import cosmology as cosmo

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
            galaxies=[source_galaxy], cosmology=cosmo.FLRW, phase_name="test_phase"
        )

        assert phase_imaging_7x7.meta_dataset.pixelization is None
        assert phase_imaging_7x7.meta_dataset.has_pixelization is False
        assert phase_imaging_7x7.meta_dataset.pixelizaition_is_model == False

        source_galaxy = ag.Galaxy(
            redshift=0.5,
            pixelization=ag.pix.Rectangular(),
            regularization=ag.reg.Constant(),
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=[source_galaxy], cosmology=cosmo.FLRW, phase_name="test_phase"
        )

        assert isinstance(
            phase_imaging_7x7.meta_dataset.pixelization, ag.pix.Rectangular
        )
        assert phase_imaging_7x7.meta_dataset.has_pixelization is True
        assert phase_imaging_7x7.meta_dataset.pixelizaition_is_model == False

        source_galaxy = ag.GalaxyModel(
            redshift=0.5,
            pixelization=ag.pix.Rectangular,
            regularization=ag.reg.Constant,
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=[source_galaxy], cosmology=cosmo.FLRW, phase_name="test_phase"
        )

        assert type(phase_imaging_7x7.meta_dataset.pixelization) == type(
            ag.pix.Rectangular
        )
        assert phase_imaging_7x7.meta_dataset.has_pixelization is True
        assert phase_imaging_7x7.meta_dataset.pixelizaition_is_model == True

    def test__check_if_phase_uses_cluster_inversion(self):
        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5), source=ag.GalaxyModel(redshift=1.0)
            ),
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
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is True

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5), source=ag.GalaxyModel(redshift=1.0)
            ),
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
        )

        assert phase_imaging_7x7.meta_dataset.uses_cluster_inversion is True

    def test__inversion_pixel_limit_computed_via_config_or_input(self,):
        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="phase_imaging_7x7", inversion_pixel_limit=None
        )

        assert phase_imaging_7x7.meta_dataset.inversion_pixel_limit == 3000

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="phase_imaging_7x7", inversion_pixel_limit=10
        )

        assert phase_imaging_7x7.meta_dataset.inversion_pixel_limit == 10

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="phase_imaging_7x7", inversion_pixel_limit=2000
        )

        assert phase_imaging_7x7.meta_dataset.inversion_pixel_limit == 2000
