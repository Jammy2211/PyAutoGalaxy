from os import path

import autofit as af
import autogalaxy as ag
import numpy as np
import pytest
from test_autogalaxy.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestVariants:
    def test__masked_imaging_signal_to_noise_limit(self, imaging_7x7, mask_7x7_1_pix):
        imaging_snr_limit = imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=1.0
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="phase_imaging_7x7",
            settings=ag.PhaseSettingsImaging(signal_to_noise_limit=1.0),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7_1_pix,
            results=mock_pipeline.MockResults(),
        )
        assert (
            analysis.masked_dataset.image.in_2d
            == imaging_snr_limit.image.in_2d * np.invert(mask_7x7_1_pix)
        ).all()
        assert (
            analysis.masked_dataset.noise_map.in_2d
            == imaging_snr_limit.noise_map.in_2d * np.invert(mask_7x7_1_pix)
        ).all()

        imaging_snr_limit = imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
            signal_to_noise_limit=0.1
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="phase_imaging_7x7",
            settings=ag.PhaseSettingsImaging(signal_to_noise_limit=0.1),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7_1_pix,
            results=mock_pipeline.MockResults(),
        )
        assert (
            analysis.masked_dataset.image.in_2d
            == imaging_snr_limit.image.in_2d * np.invert(mask_7x7_1_pix)
        ).all()
        assert (
            analysis.masked_dataset.noise_map.in_2d
            == imaging_snr_limit.noise_map.in_2d * np.invert(mask_7x7_1_pix)
        ).all()

    def test__masked_imaging_is_binned_up(self, imaging_7x7, mask_7x7_1_pix):
        binned_up_imaging = imaging_7x7.binned_from_bin_up_factor(bin_up_factor=2)

        binned_up_mask = mask_7x7_1_pix.mapping.binned_mask_from_bin_up_factor(
            bin_up_factor=2
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="phase_imaging_7x7",
            settings=ag.PhaseSettingsImaging(bin_up_factor=2),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7,
            mask=mask_7x7_1_pix,
            results=mock_pipeline.MockResults(),
        )
        assert (
            analysis.masked_dataset.image.in_2d
            == binned_up_imaging.image.in_2d * np.invert(binned_up_mask)
        ).all()

        assert (
            analysis.masked_dataset.psf == (1.0 / 9.0) * binned_up_imaging.psf
        ).all()
        assert (
            analysis.masked_dataset.noise_map.in_2d
            == binned_up_imaging.noise_map.in_2d * np.invert(binned_up_mask)
        ).all()

        assert (analysis.masked_dataset.mask == binned_up_mask).all()

    def test__phase_can_receive_hyper_image_and_noise_maps(self):
        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=ag.Redshift),
                galaxy1=ag.GalaxyModel(redshift=ag.Redshift),
            ),
            hyper_image_sky=ag.hyper_data.HyperImageSky,
            hyper_background_noise=ag.hyper_data.HyperBackgroundNoise,
            non_linear_class=af.MultiNest,
            phase_name="test_phase",
        )

        instance = phase_imaging_7x7.model.instance_from_vector([0.1, 0.2, 0.3, 0.4])

        assert instance.galaxies[0].redshift == 0.1
        assert instance.galaxies[1].redshift == 0.2
        assert instance.hyper_image_sky.sky_scale == 0.3
        assert instance.hyper_background_noise.noise_scale == 0.4
