from os import path

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


def test__masked_imaging__settings_inputs_are_used_in_masked_imaging(
    imaging_7x7, mask_7x7
):

    phase_imaging_7x7 = ag.PhaseImaging(
        phase_name="phase_imaging_7x7",
        settings=ag.PhaseSettingsImaging(
            grid_class=ag.Grid,
            grid_inversion_class=ag.Grid,
            sub_size=3,
            signal_to_noise_limit=1.0,
            bin_up_factor=2,
            inversion_pixel_limit=100,
            psf_shape_2d=(3, 3),
        ),
        search=mock_pipeline.MockSearch(),
    )

    assert phase_imaging_7x7.meta_dataset.settings.sub_size == 3
    assert phase_imaging_7x7.meta_dataset.settings.signal_to_noise_limit == 1.0
    assert phase_imaging_7x7.meta_dataset.settings.bin_up_factor == 2
    assert phase_imaging_7x7.meta_dataset.settings.inversion_pixel_limit == 100
    assert phase_imaging_7x7.meta_dataset.settings.psf_shape_2d == (3, 3)

    analysis = phase_imaging_7x7.make_analysis(
        dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    assert isinstance(analysis.masked_dataset.grid, ag.Grid)
    assert isinstance(analysis.masked_dataset.grid_inversion, ag.Grid)

    phase_imaging_7x7 = ag.PhaseImaging(
        phase_name="phase_imaging_7x7",
        settings=ag.PhaseSettingsImaging(
            grid_class=ag.GridIterate,
            sub_size=3,
            fractional_accuracy=0.99,
            sub_steps=[2],
        ),
        search=mock_pipeline.MockSearch(),
    )

    analysis = phase_imaging_7x7.make_analysis(
        dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    assert isinstance(analysis.masked_dataset.grid, ag.GridIterate)
    assert analysis.masked_dataset.grid.sub_size == 1
    assert analysis.masked_dataset.grid.fractional_accuracy == 0.99
    assert analysis.masked_dataset.grid.sub_steps == [2]


def test__masked_imaging__uses_signal_to_noise_limit(imaging_7x7, mask_7x7_1_pix):
    imaging_snr_limit = imaging_7x7.signal_to_noise_limited_from_signal_to_noise_limit(
        signal_to_noise_limit=1.0
    )

    phase_imaging_7x7 = ag.PhaseImaging(
        phase_name="phase_imaging_7x7",
        settings=ag.PhaseSettingsImaging(signal_to_noise_limit=1.0),
        search=mock_pipeline.MockSearch(),
    )

    analysis = phase_imaging_7x7.make_analysis(
        dataset=imaging_7x7, mask=mask_7x7_1_pix, results=mock_pipeline.MockResults()
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
        search=mock_pipeline.MockSearch(),
    )

    analysis = phase_imaging_7x7.make_analysis(
        dataset=imaging_7x7, mask=mask_7x7_1_pix, results=mock_pipeline.MockResults()
    )
    assert (
        analysis.masked_dataset.image.in_2d
        == imaging_snr_limit.image.in_2d * np.invert(mask_7x7_1_pix)
    ).all()
    assert (
        analysis.masked_dataset.noise_map.in_2d
        == imaging_snr_limit.noise_map.in_2d * np.invert(mask_7x7_1_pix)
    ).all()


def test__masked_imaging__uses_bin_up_factor(imaging_7x7, mask_7x7_1_pix):
    binned_up_imaging = imaging_7x7.binned_from_bin_up_factor(bin_up_factor=2)

    binned_up_mask = mask_7x7_1_pix.binned_mask_from_bin_up_factor(bin_up_factor=2)

    phase_imaging_7x7 = ag.PhaseImaging(
        phase_name="phase_imaging_7x7",
        settings=ag.PhaseSettingsImaging(bin_up_factor=2),
        search=mock_pipeline.MockSearch(),
    )

    analysis = phase_imaging_7x7.make_analysis(
        dataset=imaging_7x7, mask=mask_7x7_1_pix, results=mock_pipeline.MockResults()
    )
    assert (
        analysis.masked_dataset.image.in_2d
        == binned_up_imaging.image.in_2d * np.invert(binned_up_mask)
    ).all()

    assert (analysis.masked_dataset.psf == (1.0 / 9.0) * binned_up_imaging.psf).all()
    assert (
        analysis.masked_dataset.noise_map.in_2d
        == binned_up_imaging.noise_map.in_2d * np.invert(binned_up_mask)
    ).all()

    assert (analysis.masked_dataset.mask == binned_up_mask).all()


def test__phase_can_receive_hyper_image_and_noise_maps():
    phase_imaging_7x7 = ag.PhaseImaging(
        phase_name="test_phase",
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=ag.Redshift),
            galaxy1=ag.GalaxyModel(redshift=ag.Redshift),
        ),
        hyper_image_sky=ag.hyper_data.HyperImageSky,
        hyper_background_noise=ag.hyper_data.HyperBackgroundNoise,
        search=mock_pipeline.MockSearch(),
    )

    instance = phase_imaging_7x7.model.instance_from_vector([0.1, 0.2, 0.3, 0.4])

    assert instance.galaxies[0].redshift == 0.1
    assert instance.galaxies[1].redshift == 0.2
    assert instance.hyper_image_sky.sky_scale == 0.3
    assert instance.hyper_background_noise.noise_scale == 0.4
