from autoconf import conf
from autoarray.structures import grids
from autoarray.operators import transformer

import copy


class AbstractPhaseSettings:
    def __init__(
        self,
        grid_class=grids.GridIterate,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
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
        self.pixel_scales_interp = pixel_scales_interp
        self.signal_to_noise_limit = signal_to_noise_limit
        self.bin_up_factor = bin_up_factor
        self.inversion_pixel_limit = inversion_pixel_limit or conf.instance.general.get(
            "inversion", "inversion_pixel_limit_overall", int
        )

    def edit(
        self,
        grid_class=None,
        grid_inversion_class=None,
        sub_size=None,
        fractional_accuracy=None,
        sub_steps=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
    ):

        settings = copy.copy(self)

        settings.grid_class = self.grid_class if grid_class is None else grid_class
        settings.grid_inversion_class = (
            self.grid_inversion_class
            if grid_inversion_class is None
            else grid_inversion_class
        )
        settings.sub_size = self.sub_size if sub_size is None else sub_size
        settings.fractional_accuracy = (
            self.fractional_accuracy
            if fractional_accuracy is None
            else fractional_accuracy
        )
        settings.sub_steps = self.sub_steps if sub_steps is None else sub_steps
        settings.signal_to_noise_limit = (
            self.signal_to_noise_limit
            if signal_to_noise_limit is None
            else signal_to_noise_limit
        )
        settings.bin_up_factor = (
            self.bin_up_factor if bin_up_factor is None else bin_up_factor
        )
        settings.inversion_pixel_limit = (
            self.inversion_pixel_limit
            if inversion_pixel_limit is None
            else inversion_pixel_limit
        )

        return settings

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
    def pixel_scales_interp_tag(self):
        """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the interpolation \
        grid that deflection angles are computed on before interpolating to the and sub aa.

        This changes the phase name 'phase_name' as follows:

        pixel_scales_interp = 1 -> phase_name
        pixel_scales_interp = 2 -> phase_name_pixel_scales_interp_2
        pixel_scales_interp = 2 -> phase_name_pixel_scales_interp_2
        """
        if self.pixel_scales_interp is None:
            return ""
        return "__interp_{0:.3f}".format(self.pixel_scales_interp)


class PhaseSettingsImaging(AbstractPhaseSettings):
    def __init__(
        self,
        grid_class=grids.GridIterate,
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

    def edit(
        self,
        grid_class=None,
        grid_inversion_class=None,
        sub_size=None,
        fractional_accuracy=None,
        sub_steps=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        psf_shape_2d=None,
    ):

        settings = super().edit(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
        )

        settings.psf_shape_2d = (
            self.psf_shape_2d if psf_shape_2d is None else psf_shape_2d
        )

        return settings

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
        grid_class=grids.GridIterate,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        transformer_class=transformer.TransformerNUFFT,
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
        self.primary_beam_shape_2d = primary_beam_shape_2d

    def edit(
        self,
        grid_class=None,
        grid_inversion_class=None,
        sub_size=None,
        fractional_accuracy=None,
        sub_steps=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        transformer_class=None,
        primary_beam_shape_2d=None,
    ):

        settings = super().edit(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
        )

        settings.transformer_class = (
            self.transformer_class if transformer_class is None else transformer_class
        )
        settings.primary_beam_shape_2d = (
            self.primary_beam_shape_2d
            if primary_beam_shape_2d is None
            else primary_beam_shape_2d
        )

        return settings

    @property
    def phase_tag(self):

        return (
            "phase_tag"
            + self.transformer_tag
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
