from autoconf import conf
from autoarray.structures import grids
from autoarray.operators import transformer


class AbstractPhaseSettings:
    def __init__(
        self,
        grid_class=grids.GridIterator,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        interpolation_pixel_scale=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
    ):

        if sub_steps is None:
            sub_steps = [2, 4, 8, 16]

        self.grid_class = grid_class
        self.grid_inversion_class = grid_inversion_class
        self.sub_size = sub_size
        self.fractional_accuracy = fractional_accuracy
        self.sub_steps = sub_steps
        self.interpolation_pixel_scale = interpolation_pixel_scale
        self.signal_to_noise_limit = signal_to_noise_limit
        self.bin_up_factor = bin_up_factor
        self.inversion_pixel_limit = inversion_pixel_limit or conf.instance.general.get(
            "inversion", "inversion_pixel_limit_overall", int
        )

    @property
    def sub_size_tag(self):
        """Generate a sub-grid tag, to customize phase names based on the sub-grid size used.

        This changes the phase name 'phase_name' as follows:

        sub_size = None -> phase_name
        sub_size = 1 -> phase_name_sub_size_2
        sub_size = 4 -> phase_name_sub_size_4
        """
        return "__sub_" + str(self.sub_size)

    @property
    def signal_to_noise_limit_tag(self):
        """Generate a signal to noise limit tag, to customize phase names based on limiting the signal to noise ratio of
        the dataset being fitted.

        This changes the phase name 'phase_name' as follows:

        signal_to_noise_limit = None -> phase_name
        signal_to_noise_limit = 2 -> phase_name_snr_2
        signal_to_noise_limit = 10 -> phase_name_snr_10
        """
        if self.signal_to_noise_limit is None:
            return ""
        return "__snr_" + str(self.signal_to_noise_limit)

    @property
    def bin_up_factor_tag(self):
        """Generate a bin up tag, to customize phase names based on the resolutioon the image is binned up by for faster \
        run times.

        This changes the phase name 'phase_name' as follows:

        bin_up_factor = 1 -> phase_name
        bin_up_factor = 2 -> phase_name_bin_up_factor_2
        bin_up_factor = 2 -> phase_name_bin_up_factor_2
        """
        if self.bin_up_factor == 1 or self.bin_up_factor is None:
            return ""
        return "__bin_" + str(self.bin_up_factor)

    @property
    def interpolation_pixel_scale_tag(self):
        """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the interpolation \
        grid that deflection angles are computed on before interpolating to the and sub aa.

        This changes the phase name 'phase_name' as follows:

        interpolation_pixel_scale = 1 -> phase_name
        interpolation_pixel_scale = 2 -> phase_name_interpolation_pixel_scale_2
        interpolation_pixel_scale = 2 -> phase_name_interpolation_pixel_scale_2
        """
        if self.interpolation_pixel_scale is None:
            return ""
        return "__interp_{0:.3f}".format(self.interpolation_pixel_scale)


class PhaseSettingsImaging(AbstractPhaseSettings):
    def __init__(
        self,
        grid_class=grids.GridIterator,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        psf_shape_2d=None,
    ):

        super().__init__(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
        )

        self.psf_shape_2d = psf_shape_2d

    @property
    def phase_tag(self,):
        return (
            "phase_tag"
            + self.sub_size_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.psf_shape_tag
        )

    @property
    def psf_shape_tag(self):
        """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
        is trimmed to for faster run times.

        This changes the phase name 'phase_name' as follows:

        image_psf_shape = 1 -> phase_name
        image_psf_shape = 2 -> phase_name_image_psf_shape_2
        image_psf_shape = 2 -> phase_name_image_psf_shape_2
        """
        if self.psf_shape_2d is None:
            return ""
        y = str(self.psf_shape_2d[0])
        x = str(self.psf_shape_2d[1])
        return "__psf_" + y + "x" + x


class PhaseSettingsInterferometer(AbstractPhaseSettings):
    def __init__(
        self,
        grid_class=grids.GridIterator,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        transformer_class=transformer.TransformerNUFFT,
        real_space_shape_2d=None,
        real_space_pixel_scales=None,
        primary_beam_shape_2d=None,
    ):

        super().__init__(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
        )

        self.transformer_class = transformer_class
        self.real_space_shape_2d = real_space_shape_2d
        self.real_space_pixel_scales = real_space_pixel_scales
        self.primary_beam_shape_2d = primary_beam_shape_2d

    @property
    def phase_tag(self):

        return (
            "phase_tag"
            + self.transformer_tag
            + self.real_space_shape_2d_tag
            + self.real_space_pixel_scales_tag
            + self.sub_size_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.primary_beam_shape_tag
        )

    @property
    def transformer_tag(self):
        """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
        is trimmed to for faster run times.

        This changes the phase name 'phase_name' as follows:

        image_psf_shape = 1 -> phase_name
        image_psf_shape = 2 -> phase_name_image_psf_shape_2
        image_psf_shape = 2 -> phase_name_image_psf_shape_2
        """
        if self.transformer_class is transformer.TransformerDFT:
            return "__dft"
        elif self.transformer_class is transformer.TransformerFFT:
            return "__fft"
        elif self.transformer_class is transformer.TransformerNUFFT:
            return "__nufft"
        elif self.transformer_class is None:
            return ""

    @property
    def real_space_shape_2d_tag(self):
        """Generate a sub-grid tag, to customize phase names based on the sub-grid size used.

        This changes the phase name 'phase_name' as follows:

        real_space_shape_2d = None -> phase_name
        real_space_shape_2d = 1 -> phase_name_real_space_shape_2d_2
        real_space_shape_2d = 4 -> phase_name_real_space_shape_2d_4
        """
        if self.real_space_shape_2d is None:
            return ""
        y = str(self.real_space_shape_2d[0])
        x = str(self.real_space_shape_2d[1])
        return "__rs_shape_" + y + "x" + x

    @property
    def real_space_pixel_scales_tag(self):
        """Generate a sub-grid tag, to customize phase names based on the sub-grid size used.

        This changes the phase name 'phase_name' as follows:

        real_space_pixel_scales = None -> phase_name
        real_space_pixel_scales = 1 -> phase_name_real_space_pixel_scales_2
        real_space_pixel_scales = 4 -> phase_name_real_space_pixel_scales_4
        """
        if self.real_space_pixel_scales is None:
            return ""
        y = "{0:.2f}".format(self.real_space_pixel_scales[0])
        x = "{0:.2f}".format(self.real_space_pixel_scales[1])
        return "__rs_pix_" + y + "x" + x

    @property
    def primary_beam_shape_tag(self):
        """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
        is trimmed to for faster run times.

        This changes the phase name 'phase_name' as follows:

        image_psf_shape = 1 -> phase_name
        image_psf_shape = 2 -> phase_name_image_psf_shape_2
        image_psf_shape = 2 -> phase_name_image_psf_shape_2
        """
        if self.primary_beam_shape_2d is None:
            return ""
        y = str(self.primary_beam_shape_2d[0])
        x = str(self.primary_beam_shape_2d[1])
        return "__pb_" + y + "x" + x
