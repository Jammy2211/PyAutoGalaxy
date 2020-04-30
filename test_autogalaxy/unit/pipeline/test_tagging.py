import autogalaxy as ag


class TestPhaseTag:
    def test__mixture_of_values(self):

        phase_tag = ag.tagging.phase_tag_from_phase_settings(
            sub_size=2, signal_to_noise_limit=2, bin_up_factor=None, psf_shape_2d=None
        )

        assert phase_tag == "phase_tag__sub_2__snr_2"

        phase_tag = ag.tagging.phase_tag_from_phase_settings(
            sub_size=1, signal_to_noise_limit=None, bin_up_factor=3, psf_shape_2d=(2, 2)
        )

        assert phase_tag == "phase_tag__sub_1__bin_3__psf_2x2"

        phase_tag = ag.tagging.phase_tag_from_phase_settings(
            sub_size=1,
            transformer_class=ag.TransformerDFT,
            real_space_shape_2d=(3, 3),
            real_space_pixel_scales=(1.0, 2.0),
            primary_beam_shape_2d=(2, 2),
        )

        assert (
            phase_tag == "phase_tag__dft__rs_shape_3x3__rs_pix_1.00x2.00__sub_1__pb_2x2"
        )


class TestPhaseTaggers:
    def test__sub_size_tagger(self):

        tag = ag.tagging.sub_size_tag_from_sub_size(sub_size=1)
        assert tag == "__sub_1"
        tag = ag.tagging.sub_size_tag_from_sub_size(sub_size=2)
        assert tag == "__sub_2"
        tag = ag.tagging.sub_size_tag_from_sub_size(sub_size=4)
        assert tag == "__sub_4"

    def test__signal_to_noise_limit_tagger(self):

        tag = ag.tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=None
        )
        assert tag == ""
        tag = ag.tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=1
        )
        assert tag == "__snr_1"
        tag = ag.tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=2
        )
        assert tag == "__snr_2"
        tag = ag.tagging.signal_to_noise_limit_tag_from_signal_to_noise_limit(
            signal_to_noise_limit=3
        )
        assert tag == "__snr_3"

    def test__bin_up_factor_tagger(self):

        tag = ag.tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=None)
        assert tag == ""
        tag = ag.tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=1)
        assert tag == ""
        tag = ag.tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=2)
        assert tag == "__bin_2"
        tag = ag.tagging.bin_up_factor_tag_from_bin_up_factor(bin_up_factor=3)
        assert tag == "__bin_3"

    def test__psf_shape_2d_tagger(self):

        tag = ag.tagging.psf_shape_tag_from_psf_shape_2d(psf_shape_2d=None)
        assert tag == ""
        tag = ag.tagging.psf_shape_tag_from_psf_shape_2d(psf_shape_2d=(2, 2))
        assert tag == "__psf_2x2"
        tag = ag.tagging.psf_shape_tag_from_psf_shape_2d(psf_shape_2d=(3, 4))
        assert tag == "__psf_3x4"

    def test__transformer_tagger(self):
        tag = ag.tagging.transformer_tag_from_transformer_class(
            transformer_class=ag.TransformerDFT
        )
        assert tag == "__dft"
        tag = ag.tagging.transformer_tag_from_transformer_class(
            transformer_class=ag.TransformerFFT
        )
        assert tag == "__fft"
        tag = ag.tagging.transformer_tag_from_transformer_class(
            transformer_class=ag.TransformerNUFFT
        )
        assert tag == "__nufft"
        tag = ag.tagging.transformer_tag_from_transformer_class(transformer_class=None)
        assert tag == ""

    def test__primary_beam_shape_2d_tagger(self):
        tag = ag.tagging.primary_beam_shape_tag_from_primary_beam_shape_2d(
            primary_beam_shape_2d=None
        )
        assert tag == ""
        tag = ag.tagging.primary_beam_shape_tag_from_primary_beam_shape_2d(
            primary_beam_shape_2d=(2, 2)
        )
        assert tag == "__pb_2x2"
        tag = ag.tagging.primary_beam_shape_tag_from_primary_beam_shape_2d(
            primary_beam_shape_2d=(3, 4)
        )
        assert tag == "__pb_3x4"

    def test__real_space_shape_2d_tagger(self):

        tag = ag.tagging.real_space_shape_2d_tag_from_real_space_shape_2d(
            real_space_shape_2d=None
        )
        assert tag == ""
        tag = ag.tagging.real_space_shape_2d_tag_from_real_space_shape_2d(
            real_space_shape_2d=(2, 2)
        )
        assert tag == "__rs_shape_2x2"
        tag = ag.tagging.real_space_shape_2d_tag_from_real_space_shape_2d(
            real_space_shape_2d=(3, 4)
        )
        assert tag == "__rs_shape_3x4"

    def test__real_space_pixel_scales_tagger(self):

        tag = ag.tagging.real_space_pixel_scales_tag_from_real_space_pixel_scales(
            real_space_pixel_scales=None
        )
        assert tag == ""
        tag = ag.tagging.real_space_pixel_scales_tag_from_real_space_pixel_scales(
            real_space_pixel_scales=(0.01, 0.02)
        )
        assert tag == "__rs_pix_0.01x0.02"
        tag = ag.tagging.real_space_pixel_scales_tag_from_real_space_pixel_scales(
            real_space_pixel_scales=(2.0, 1.0)
        )
        assert tag == "__rs_pix_2.00x1.00"
