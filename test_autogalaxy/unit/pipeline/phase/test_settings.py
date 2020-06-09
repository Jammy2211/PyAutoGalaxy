import autogalaxy as ag


class TestTags:
    def test__grids__sub_size_tags(self):

        settings = ag.PhaseSettingsImaging(grid_class=ag.GridIterate, sub_size=1)
        assert settings.grid_sub_size_tag == ""
        settings = ag.PhaseSettingsImaging(grid_class=ag.Grid, sub_size=1)
        assert settings.grid_sub_size_tag == "sub_1"
        settings = ag.PhaseSettingsImaging(grid_class=ag.Grid, sub_size=2)
        assert settings.grid_sub_size_tag == "sub_2"
        settings = ag.PhaseSettingsImaging(grid_class=ag.Grid, sub_size=4)
        assert settings.grid_sub_size_tag == "sub_4"

        settings = ag.PhaseSettingsImaging(
            grid_inversion_class=ag.GridIterate, sub_size=1
        )
        assert settings.grid_inversion_sub_size_tag == ""
        settings = ag.PhaseSettingsImaging(grid_inversion_class=ag.Grid, sub_size=1)
        assert settings.grid_inversion_sub_size_tag == "sub_1"
        settings = ag.PhaseSettingsImaging(grid_inversion_class=ag.Grid, sub_size=2)
        assert settings.grid_inversion_sub_size_tag == "sub_2"
        settings = ag.PhaseSettingsImaging(grid_inversion_class=ag.Grid, sub_size=4)
        assert settings.grid_inversion_sub_size_tag == "sub_4"

    def test__grids__fractional_accuracy_tags(self):

        settings = ag.PhaseSettingsImaging(grid_class=ag.Grid, fractional_accuracy=1)
        assert settings.grid_fractional_accuracy_tag == ""
        settings = ag.PhaseSettingsImaging(
            grid_class=ag.GridIterate, fractional_accuracy=0.5
        )
        assert settings.grid_fractional_accuracy_tag == "facc_0.5"
        settings = ag.PhaseSettingsImaging(
            grid_class=ag.GridIterate, fractional_accuracy=0.71
        )
        assert settings.grid_fractional_accuracy_tag == "facc_0.71"
        settings = ag.PhaseSettingsImaging(
            grid_class=ag.GridIterate, fractional_accuracy=0.999999
        )
        assert settings.grid_fractional_accuracy_tag == "facc_0.999999"

        settings = ag.PhaseSettingsImaging(
            grid_inversion_class=ag.Grid, fractional_accuracy=1
        )
        assert settings.grid_inversion_fractional_accuracy_tag == ""
        settings = ag.PhaseSettingsImaging(
            grid_inversion_class=ag.GridIterate, fractional_accuracy=0.5
        )
        assert settings.grid_inversion_fractional_accuracy_tag == "facc_0.5"
        settings = ag.PhaseSettingsImaging(
            grid_inversion_class=ag.GridIterate, fractional_accuracy=0.71
        )
        assert settings.grid_inversion_fractional_accuracy_tag == "facc_0.71"
        settings = ag.PhaseSettingsImaging(
            grid_inversion_class=ag.GridIterate, fractional_accuracy=0.999999
        )
        assert settings.grid_inversion_fractional_accuracy_tag == "facc_0.999999"

    def test__grid__pixel_scales_interp_tag(self):

        settings = ag.PhaseSettingsImaging(grid_class=ag.Grid, pixel_scales_interp=0.1)
        assert settings.grid_pixel_scales_interp_tag == ""
        settings = ag.PhaseSettingsImaging(
            grid_class=ag.GridInterpolate, pixel_scales_interp=None
        )
        assert settings.grid_pixel_scales_interp_tag == ""
        settings = ag.PhaseSettingsImaging(
            grid_class=ag.GridInterpolate, pixel_scales_interp=0.5
        )
        assert settings.grid_pixel_scales_interp_tag == "interp_0.500"
        settings = ag.PhaseSettingsImaging(
            grid_class=ag.GridInterpolate, pixel_scales_interp=0.25
        )
        assert settings.grid_pixel_scales_interp_tag == "interp_0.250"
        settings = ag.PhaseSettingsImaging(
            grid_class=ag.GridInterpolate, pixel_scales_interp=0.234
        )
        assert settings.grid_pixel_scales_interp_tag == "interp_0.234"

        settings = ag.PhaseSettingsImaging(
            grid_inversion_class=ag.Grid, pixel_scales_interp=0.1
        )
        assert settings.grid_inversion_pixel_scales_interp_tag == ""
        settings = ag.PhaseSettingsImaging(
            grid_inversion_class=ag.GridInterpolate, pixel_scales_interp=None
        )
        assert settings.grid_inversion_pixel_scales_interp_tag == ""
        settings = ag.PhaseSettingsImaging(
            grid_inversion_class=ag.GridInterpolate, pixel_scales_interp=0.5
        )
        assert settings.grid_inversion_pixel_scales_interp_tag == "interp_0.500"
        settings = ag.PhaseSettingsImaging(
            grid_inversion_class=ag.GridInterpolate, pixel_scales_interp=0.25
        )
        assert settings.grid_inversion_pixel_scales_interp_tag == "interp_0.250"
        settings = ag.PhaseSettingsImaging(
            grid_inversion_class=ag.GridInterpolate, pixel_scales_interp=0.234
        )
        assert settings.grid_inversion_pixel_scales_interp_tag == "interp_0.234"

    def test__grid_tags(self):

        settings = ag.PhaseSettingsImaging(
            grid_class=ag.Grid,
            sub_size=1,
            grid_inversion_class=ag.GridIterate,
            fractional_accuracy=0.5,
        )
        assert settings.grid_no_inversion_tag == "__grid_sub_1"
        assert settings.grid_with_inversion_tag == "__grid_sub_1_inv_facc_0.5"

        settings = ag.PhaseSettingsImaging(
            grid_class=ag.GridInterpolate,
            grid_inversion_class=ag.GridInterpolate,
            pixel_scales_interp=0.5,
        )
        assert settings.grid_no_inversion_tag == "__grid_interp_0.500"
        assert (
            settings.grid_with_inversion_tag == "__grid_interp_0.500_inv_interp_0.500"
        )

        settings = ag.PhaseSettingsImaging(
            grid_class=ag.GridIterate,
            fractional_accuracy=0.8,
            grid_inversion_class=ag.Grid,
            sub_size=2,
        )
        assert settings.grid_no_inversion_tag == "__grid_facc_0.8"
        assert settings.grid_with_inversion_tag == "__grid_facc_0.8_inv_sub_2"

    def test__signal_to_noise_limit_tag(self):

        settings = ag.PhaseSettingsImaging(signal_to_noise_limit=None)
        assert settings.signal_to_noise_limit_tag == ""
        settings = ag.PhaseSettingsImaging(signal_to_noise_limit=1)
        assert settings.signal_to_noise_limit_tag == "__snr_1"
        settings = ag.PhaseSettingsImaging(signal_to_noise_limit=2)
        assert settings.signal_to_noise_limit_tag == "__snr_2"

    def test__bin_up_factor_tag(self):

        settings = ag.PhaseSettingsImaging(bin_up_factor=None)
        assert settings.bin_up_factor_tag == ""
        settings = ag.PhaseSettingsImaging(bin_up_factor=1)
        assert settings.bin_up_factor_tag == ""
        settings = ag.PhaseSettingsImaging(bin_up_factor=2)
        assert settings.bin_up_factor_tag == "__bin_2"

    def test__psf_shape_2d_tag(self):

        settings = ag.PhaseSettingsImaging(psf_shape_2d=None)
        assert settings.psf_shape_tag == ""
        settings = ag.PhaseSettingsImaging(psf_shape_2d=(2, 2))
        assert settings.psf_shape_tag == "__psf_2x2"
        settings = ag.PhaseSettingsImaging(psf_shape_2d=(3, 4))
        assert settings.psf_shape_tag == "__psf_3x4"

    def test__transformer_tag(self):
        settings = ag.PhaseSettingsInterferometer(transformer_class=ag.TransformerDFT)
        assert settings.transformer_tag == "__dft"
        settings = ag.PhaseSettingsInterferometer(transformer_class=ag.TransformerFFT)
        assert settings.transformer_tag == "__fft"
        settings = ag.PhaseSettingsInterferometer(transformer_class=ag.TransformerNUFFT)
        assert settings.transformer_tag == "__nufft"
        settings = ag.PhaseSettingsInterferometer(transformer_class=None)
        assert settings.transformer_tag == ""

    def test__primary_beam_shape_2d_tag(self):
        settings = ag.PhaseSettingsInterferometer(primary_beam_shape_2d=None)
        assert settings.primary_beam_shape_tag == ""
        settings = ag.PhaseSettingsInterferometer(primary_beam_shape_2d=(2, 2))
        assert settings.primary_beam_shape_tag == "__pb_2x2"
        settings = ag.PhaseSettingsInterferometer(primary_beam_shape_2d=(3, 4))
        assert settings.primary_beam_shape_tag == "__pb_3x4"

    def test__tag__mixture_of_values(self):

        settings = ag.PhaseSettingsImaging(
            grid_class=ag.Grid,
            grid_inversion_class=ag.Grid,
            sub_size=2,
            signal_to_noise_limit=2,
            bin_up_factor=None,
            psf_shape_2d=None,
        )

        assert settings.phase_no_inversion_tag == "settings__grid_sub_2__snr_2"
        assert (
            settings.phase_with_inversion_tag == "settings__grid_sub_2_inv_sub_2__snr_2"
        )

        settings = ag.PhaseSettingsImaging(
            grid_class=ag.Grid,
            grid_inversion_class=ag.GridIterate,
            sub_size=1,
            fractional_accuracy=0.1,
            signal_to_noise_limit=None,
            bin_up_factor=3,
            psf_shape_2d=(2, 2),
        )

        assert settings.phase_no_inversion_tag == "settings__grid_sub_1__bin_3__psf_2x2"
        assert (
            settings.phase_with_inversion_tag
            == "settings__grid_sub_1_inv_facc_0.1__bin_3__psf_2x2"
        )

        settings = ag.PhaseSettingsInterferometer(
            grid_class=ag.GridIterate,
            grid_inversion_class=ag.Grid,
            fractional_accuracy=0.1,
            sub_size=3,
            transformer_class=ag.TransformerDFT,
            primary_beam_shape_2d=(2, 2),
            log_likelihood_cap=200.001,
        )

        assert (
            settings.phase_no_inversion_tag
            == "settings__grid_facc_0.1__dft__pb_2x2__lh_cap_200.0"
        )
        assert (
            settings.phase_with_inversion_tag
            == "settings__grid_facc_0.1_inv_sub_3__dft__pb_2x2__lh_cap_200.0"
        )
