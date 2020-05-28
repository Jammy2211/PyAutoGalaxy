import autogalaxy as ag


class TestPhaseTag:
    def test__mixture_of_values(self):

        settings = ag.PhaseSettingsImaging(
            sub_size=2, signal_to_noise_limit=2, bin_up_factor=None, psf_shape_2d=None
        )

        assert settings.phase_tag == "phase_tag__sub_2__snr_2"

        settings = ag.PhaseSettingsImaging(
            sub_size=1, signal_to_noise_limit=None, bin_up_factor=3, psf_shape_2d=(2, 2)
        )

        assert settings.phase_tag == "phase_tag__sub_1__bin_3__psf_2x2"

        settings = ag.PhaseSettingsInterferometer(
            sub_size=1,
            transformer_class=ag.TransformerDFT,
            real_space_shape_2d=(3, 3),
            real_space_pixel_scales=(1.0, 2.0),
            primary_beam_shape_2d=(2, 2),
        )

        assert (
            settings.phase_tag
            == "phase_tag__dft__rs_shape_3x3__rs_pix_1.00x2.00__sub_1__pb_2x2"
        )


class TestPhaseTaggers:
    def test__sub_size_tagger(self):

        settings = ag.PhaseSettingsImaging(sub_size=1)
        assert settings.sub_size_tag == "__sub_1"
        settings = ag.PhaseSettingsImaging(sub_size=2)
        assert settings.sub_size_tag == "__sub_2"
        settings = ag.PhaseSettingsImaging(sub_size=4)
        assert settings.sub_size_tag == "__sub_4"

    def test__signal_to_noise_limit_tagger(self):

        settings = ag.PhaseSettingsImaging(signal_to_noise_limit=None)
        assert settings.signal_to_noise_limit_tag == ""
        settings = ag.PhaseSettingsImaging(signal_to_noise_limit=1)
        assert settings.signal_to_noise_limit_tag == "__snr_1"
        settings = ag.PhaseSettingsImaging(signal_to_noise_limit=2)
        assert settings.signal_to_noise_limit_tag == "__snr_2"

    def test__bin_up_factor_tagger(self):

        settings = ag.PhaseSettingsImaging(bin_up_factor=None)
        assert settings.bin_up_factor_tag == ""
        settings = ag.PhaseSettingsImaging(bin_up_factor=1)
        assert settings.bin_up_factor_tag == ""
        settings = ag.PhaseSettingsImaging(bin_up_factor=2)
        assert settings.bin_up_factor_tag == "__bin_2"

    def test__psf_shape_2d_tagger(self):

        settings = ag.PhaseSettingsImaging(psf_shape_2d=None)
        assert settings.psf_shape_tag == ""
        settings = ag.PhaseSettingsImaging(psf_shape_2d=(2, 2))
        assert settings.psf_shape_tag == "__psf_2x2"
        settings = ag.PhaseSettingsImaging(psf_shape_2d=(3, 4))
        assert settings.psf_shape_tag == "__psf_3x4"

    def test__transformer_tagger(self):
        settings = ag.PhaseSettingsInterferometer(transformer_class=ag.TransformerDFT)
        assert settings.transformer_tag == "__dft"
        settings = ag.PhaseSettingsInterferometer(transformer_class=ag.TransformerFFT)
        assert settings.transformer_tag == "__fft"
        settings = ag.PhaseSettingsInterferometer(transformer_class=ag.TransformerNUFFT)
        assert settings.transformer_tag == "__nufft"
        settings = ag.PhaseSettingsInterferometer(transformer_class=None)
        assert settings.transformer_tag == ""

    def test__primary_beam_shape_2d_tagger(self):
        settings = ag.PhaseSettingsInterferometer(primary_beam_shape_2d=None)
        assert settings.primary_beam_shape_tag == ""
        settings = ag.PhaseSettingsInterferometer(primary_beam_shape_2d=(2, 2))
        assert settings.primary_beam_shape_tag == "__pb_2x2"
        settings = ag.PhaseSettingsInterferometer(primary_beam_shape_2d=(3, 4))
        assert settings.primary_beam_shape_tag == "__pb_3x4"

    def test__real_space_shape_2d_tagger(self):

        settings = ag.PhaseSettingsInterferometer(real_space_shape_2d=None)
        assert settings.real_space_shape_2d_tag == ""
        settings = ag.PhaseSettingsInterferometer(real_space_shape_2d=(2, 2))
        assert settings.real_space_shape_2d_tag == "__rs_shape_2x2"
        settings = ag.PhaseSettingsInterferometer(real_space_shape_2d=(3, 4))
        assert settings.real_space_shape_2d_tag == "__rs_shape_3x4"

    def test__real_space_pixel_scales_tagger(self):

        settings = ag.PhaseSettingsInterferometer(real_space_pixel_scales=None)
        assert settings.real_space_pixel_scales_tag == ""
        settings = ag.PhaseSettingsInterferometer(real_space_pixel_scales=(0.01, 0.02))
        assert settings.real_space_pixel_scales_tag == "__rs_pix_0.01x0.02"
        settings = ag.PhaseSettingsInterferometer(real_space_pixel_scales=(2.0, 1.0))
        assert settings.real_space_pixel_scales_tag == "__rs_pix_2.00x1.00"
