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
            primary_beam_shape_2d=(2, 2),
        )

        assert settings.phase_tag == "phase_tag__dft__sub_1__pb_2x2"


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


class TestEdit:
    def test__imaging__edit__changes_settings_if_input(self):

        settings = ag.PhaseSettingsImaging(
            grid_class=ag.Grid,
            grid_inversion_class=ag.Grid,
            sub_size=2,
            fractional_accuracy=0.5,
            sub_steps=[2],
            signal_to_noise_limit=2,
            bin_up_factor=3,
            inversion_pixel_limit=100,
            psf_shape_2d=(3, 3),
        )

        assert settings.grid_class is ag.Grid
        assert settings.grid_inversion_class is ag.Grid
        assert settings.sub_size == 2
        assert settings.fractional_accuracy == 0.5
        assert settings.sub_steps == [2]
        assert settings.signal_to_noise_limit == 2
        assert settings.bin_up_factor == 3
        assert settings.inversion_pixel_limit == 100
        assert settings.psf_shape_2d == (3, 3)

        settings = settings.edit(
            grid_class=ag.GridIterator,
            grid_inversion_class=ag.GridInterpolator,
            sub_steps=[5],
            inversion_pixel_limit=200,
        )

        assert settings.grid_class is ag.GridIterator
        assert settings.grid_inversion_class is ag.GridInterpolator
        assert settings.sub_size == 2
        assert settings.fractional_accuracy == 0.5
        assert settings.sub_steps == [5]
        assert settings.signal_to_noise_limit == 2
        assert settings.bin_up_factor == 3
        assert settings.inversion_pixel_limit == 200
        assert settings.psf_shape_2d == (3, 3)

        settings = settings.edit(
            sub_size=3,
            fractional_accuracy=0.7,
            signal_to_noise_limit=4,
            bin_up_factor=5,
            psf_shape_2d=(5, 5),
        )

        assert settings.grid_class is ag.GridIterator
        assert settings.grid_inversion_class is ag.GridInterpolator
        assert settings.sub_size == 3
        assert settings.fractional_accuracy == 0.7
        assert settings.sub_steps == [5]
        assert settings.signal_to_noise_limit == 4
        assert settings.bin_up_factor == 5
        assert settings.inversion_pixel_limit == 200
        assert settings.psf_shape_2d == (5, 5)

    def test__interferometer__edit__changes_settings_if_input(self):

        settings = ag.PhaseSettingsInterferometer(
            grid_class=ag.Grid,
            grid_inversion_class=ag.Grid,
            sub_size=2,
            fractional_accuracy=0.5,
            sub_steps=[2],
            signal_to_noise_limit=2,
            bin_up_factor=3,
            inversion_pixel_limit=100,
            transformer_class=ag.TransformerDFT,
            primary_beam_shape_2d=(3, 3),
        )

        assert settings.grid_class is ag.Grid
        assert settings.grid_inversion_class is ag.Grid
        assert settings.sub_size == 2
        assert settings.fractional_accuracy == 0.5
        assert settings.sub_steps == [2]
        assert settings.signal_to_noise_limit == 2
        assert settings.bin_up_factor == 3
        assert settings.inversion_pixel_limit == 100
        assert settings.transformer_class is ag.TransformerDFT
        assert settings.primary_beam_shape_2d == (3, 3)

        settings = settings.edit(
            grid_class=ag.GridIterator,
            grid_inversion_class=ag.GridInterpolator,
            sub_steps=[5],
            inversion_pixel_limit=200,
            transformer_class=ag.TransformerFFT,
        )

        assert settings.grid_class is ag.GridIterator
        assert settings.grid_inversion_class is ag.GridInterpolator
        assert settings.sub_size == 2
        assert settings.fractional_accuracy == 0.5
        assert settings.sub_steps == [5]
        assert settings.signal_to_noise_limit == 2
        assert settings.bin_up_factor == 3
        assert settings.inversion_pixel_limit == 200
        assert settings.transformer_class is ag.TransformerFFT
        assert settings.primary_beam_shape_2d == (3, 3)

        settings = settings.edit(
            sub_size=3,
            fractional_accuracy=0.7,
            signal_to_noise_limit=4,
            bin_up_factor=5,
            primary_beam_shape_2d=(5, 5),
        )

        assert settings.grid_class is ag.GridIterator
        assert settings.grid_inversion_class is ag.GridInterpolator
        assert settings.sub_size == 3
        assert settings.fractional_accuracy == 0.7
        assert settings.sub_steps == [5]
        assert settings.signal_to_noise_limit == 4
        assert settings.bin_up_factor == 5
        assert settings.inversion_pixel_limit == 200
        assert settings.transformer_class is ag.TransformerFFT
        assert settings.primary_beam_shape_2d == (5, 5)
