import autogalaxy as ag


def test__tag__mixture_of_values():

    settings = ag.PhaseSettingsImaging(
        masked_imaging_settings=ag.MaskedImagingSettings(
            grid_class=ag.Grid,
            grid_inversion_class=ag.Grid,
            sub_size=2,
            signal_to_noise_limit=2,
            bin_up_factor=None,
            psf_shape_2d=None,
        )
    )

    assert settings.phase_no_inversion_tag == "settings__grid_sub_2__snr_2"
    assert settings.phase_with_inversion_tag == "settings__grid_sub_2_inv_sub_2__snr_2"

    settings = ag.PhaseSettingsImaging(
        masked_imaging_settings=ag.MaskedImagingSettings(
            grid_class=ag.Grid,
            grid_inversion_class=ag.GridIterate,
            sub_size=1,
            fractional_accuracy=0.1,
            signal_to_noise_limit=None,
            bin_up_factor=3,
            psf_shape_2d=(2, 2),
        )
    )

    assert settings.phase_no_inversion_tag == "settings__grid_sub_1__bin_3__psf_2x2"
    assert (
        settings.phase_with_inversion_tag
        == "settings__grid_sub_1_inv_facc_0.1__bin_3__psf_2x2"
    )

    settings = ag.PhaseSettingsInterferometer(
        masked_interferometer_settings=ag.MaskedInterferometerSettings(
            grid_class=ag.GridIterate,
            grid_inversion_class=ag.Grid,
            fractional_accuracy=0.1,
            sub_size=3,
            transformer_class=ag.TransformerDFT,
        ),
        inversion_settings=ag.InversionSettings(use_linear_operators=False),
        log_likelihood_cap=200.001,
    )

    assert (
        settings.phase_no_inversion_tag == "settings__grid_facc_0.1__dft__lh_cap_200.0"
    )
    assert (
        settings.phase_with_inversion_tag
        == "settings__grid_facc_0.1_inv_sub_3__dft__lh_cap_200.0"
    )

    settings = ag.PhaseSettingsInterferometer(
        masked_interferometer_settings=ag.MaskedInterferometerSettings(
            grid_class=ag.GridIterate,
            grid_inversion_class=ag.Grid,
            fractional_accuracy=0.1,
            sub_size=3,
            transformer_class=ag.TransformerNUFFT,
        ),
        inversion_settings=ag.InversionSettings(use_linear_operators=True),
    )

    assert settings.phase_no_inversion_tag == "settings__grid_facc_0.1__nufft"
    assert (
        settings.phase_with_inversion_tag
        == "settings__grid_facc_0.1_inv_sub_3__nufft__lop"
    )
