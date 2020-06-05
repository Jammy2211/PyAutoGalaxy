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
        pixel_scales_interp=None,
        sub_steps=None,
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

    @property
    def grid_no_inversion_tag(self):
        """Generate a tag describing the the grid and grid_inversions used by the phase.

        This assumes both grids were used in the analysis.
        """

        return (
            "__"
            + conf.instance.tag.get("phase", "grid", str)
            + "_"
            + self.grid_sub_size_tag
            + self.grid_fractional_accuracy_tag
            + self.grid_pixel_scales_interp_tag
        )

    @property
    def grid_with_inversion_tag(self):
        """Generate a tag describing the the grid and grid_inversions used by the phase.

        This assumes both grids were used in the analysis.
        """
        return (
            "__"
            + conf.instance.tag.get("phase", "grid", str)
            + "_"
            + self.grid_sub_size_tag
            + self.grid_fractional_accuracy_tag
            + self.grid_pixel_scales_interp_tag
            + "_"
            + conf.instance.tag.get("phase", "grid_inversion", str)
            + "_"
            + self.grid_inversion_sub_size_tag
            + self.grid_inversion_fractional_accuracy_tag
            + self.grid_inversion_pixel_scales_interp_tag
        )

    @property
    def grid_sub_size_tag(self):
        """Generate a sub-size tag, to customize phase names based on the sub-grid size used, of the Grid class.

        This changes the phase settings folder is tagged as follows:

        sub_size = None -> settings
        sub_size = 1 -> settings_sub_size_2
        sub_size = 4 -> settings_sub_size_4
        """
        if not self.grid_class is grids.Grid:
            return ""
        return (
            conf.instance.tag.get("phase", "sub_size", str) + "_" + str(self.sub_size)
        )

    @property
    def grid_fractional_accuracy_tag(self):
        """Generate a fractional accuracy tag, to customize phase names based on the fractional accuracy of the
        GridIterate class.

        This changes the phase settings folder is tagged as follows:

        fraction_accuracy = 0.5 -> settings__facc_0.5
        fractional_accuracy = 0.999999 = 4 -> settings__facc_0.999999
        """
        if not self.grid_class is grids.GridIterate:
            return ""
        return (
            conf.instance.tag.get("phase", "fractional_accuracy", str)
            + "_"
            + str(self.fractional_accuracy)
        )

    @property
    def grid_pixel_scales_interp_tag(self):
        """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the 
        GridInterpolate.

        This changes the phase settings folder is tagged as follows:

        pixel_scales_interp = None -> settings
        pixel_scales_interp = 0.1 -> settings___grid_interp_0.1
        """
        if not self.grid_class is grids.GridInterpolate:
            return ""
        if self.pixel_scales_interp is None:
            return ""
        return conf.instance.tag.get(
            "phase", "pixel_scales_interp", str
        ) + "_{0:.3f}".format(self.pixel_scales_interp)

    @property
    def grid_inversion_sub_size_tag(self):
        """Generate a sub-size tag, to customize phase names based on the sub-grid size used, of the Grid class.

        This changes the phase settings folder is tagged as follows:

        sub_size = None -> settings
        sub_size = 1 -> settings__grid_sub_size_2
        sub_size = 4 -> settings__grid_inv_sub_size_4
        """
        if not self.grid_inversion_class is grids.Grid:
            return ""
        return (
            conf.instance.tag.get("phase", "sub_size", str) + "_" + str(self.sub_size)
        )

    @property
    def grid_inversion_fractional_accuracy_tag(self):
        """Generate a fractional accuracy tag, to customize phase names based on the fractional accuracy of the
        GridIterate class.

        This changes the phase settings folder is tagged as follows:

        fraction_accuracy = 0.5 -> settings__facc_0.5
        fractional_accuracy = 0.999999 = 4 -> settings__facc_0.999999
        """
        if not self.grid_inversion_class is grids.GridIterate:
            return ""
        return (
            conf.instance.tag.get("phase", "fractional_accuracy", str)
            + "_"
            + str(self.fractional_accuracy)
        )

    @property
    def grid_inversion_pixel_scales_interp_tag(self):
        """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the 
        GridInterpolate.

        This changes the phase settings folder is tagged as follows:

        pixel_scales_interp = None -> settings
        pixel_scales_interp = 0.1 -> settings___grid_interp_0.1
        """
        if not self.grid_inversion_class is grids.GridInterpolate:
            return ""
        if self.pixel_scales_interp is None:
            return ""
        return conf.instance.tag.get(
            "phase", "pixel_scales_interp", str
        ) + "_{0:.3f}".format(self.pixel_scales_interp)

    @property
    def signal_to_noise_limit_tag(self):
        """Generate a signal to noise limit tag, to customize phase names based on limiting the signal to noise ratio of
        the dataset being fitted.

        This changes the phase settings folder is tagged as follows:

        signal_to_noise_limit = None -> settings
        signal_to_noise_limit = 2 -> settings_snr_2
        signal_to_noise_limit = 10 -> settings_snr_10
        """
        if self.signal_to_noise_limit is None:
            return ""
        return (
            "__"
            + conf.instance.tag.get("phase", "signal_to_noise_limit", str)
            + "_"
            + str(self.signal_to_noise_limit)
        )

    @property
    def bin_up_factor_tag(self):
        """Generate a bin up tag, to customize phase names based on the resolutioon the image is binned up by for faster \
        run times.

        This changes the phase settings folder is tagged as follows:

        bin_up_factor = 1 -> settings
        bin_up_factor = 2 -> settings_bin_up_factor_2
        bin_up_factor = 2 -> settings_bin_up_factor_2
        """
        if self.bin_up_factor == 1 or self.bin_up_factor is None:
            return ""
        return (
            "__"
            + conf.instance.tag.get("phase", "bin_up_factor", str)
            + "_"
            + str(self.bin_up_factor)
        )


class PhaseSettingsImaging(AbstractPhaseSettings):
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
        psf_shape_2d=None,
    ):

        super().__init__(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            pixel_scales_interp=pixel_scales_interp,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
        )

        self.psf_shape_2d = psf_shape_2d

    @property
    def phase_no_inversion_tag(self):
        return (
            conf.instance.tag.get("phase", "phase", str)
            + self.grid_no_inversion_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.psf_shape_tag
        )

    @property
    def phase_with_inversion_tag(self):
        return (
            conf.instance.tag.get("phase", "phase", str)
            + self.grid_with_inversion_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.psf_shape_tag
        )

    @property
    def psf_shape_tag(self):
        """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
        is trimmed to for faster run times.

        This changes the phase settings folder is tagged as follows:

        image_psf_shape = 1 -> settings
        image_psf_shape = 2 -> settings_image_psf_shape_2
        image_psf_shape = 2 -> settings_image_psf_shape_2
        """
        if self.psf_shape_2d is None:
            return ""
        y = str(self.psf_shape_2d[0])
        x = str(self.psf_shape_2d[1])
        return (
            "__" + conf.instance.tag.get("phase", "psf_shape", str) + "_" + y + "x" + x
        )


class PhaseSettingsInterferometer(AbstractPhaseSettings):
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
        transformer_class=transformer.TransformerNUFFT,
        primary_beam_shape_2d=None,
    ):

        super().__init__(
            grid_class=grid_class,
            grid_inversion_class=grid_inversion_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            pixel_scales_interp=pixel_scales_interp,
            sub_steps=sub_steps,
            signal_to_noise_limit=signal_to_noise_limit,
            bin_up_factor=bin_up_factor,
            inversion_pixel_limit=inversion_pixel_limit,
        )

        self.transformer_class = transformer_class
        self.primary_beam_shape_2d = primary_beam_shape_2d

    @property
    def phase_no_inversion_tag(self):

        return (
            conf.instance.tag.get("phase", "phase", str)
            + self.grid_no_inversion_tag
            + self.transformer_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.primary_beam_shape_tag
        )

    @property
    def phase_with_inversion_tag(self):

        return (
            conf.instance.tag.get("phase", "phase", str)
            + self.grid_with_inversion_tag
            + self.transformer_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.primary_beam_shape_tag
        )

    @property
    def transformer_tag(self):
        """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
        is trimmed to for faster run times.

        This changes the phase settings folder is tagged as follows:

        image_psf_shape = 1 -> settings
        image_psf_shape = 2 -> settings_image_psf_shape_2
        image_psf_shape = 2 -> settings_image_psf_shape_2
        """
        if self.transformer_class is None:
            return ""
        return "__" + conf.instance.tag.get(
            "transformer", self.transformer_class.__name__, str
        )

    @property
    def primary_beam_shape_tag(self):
        """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
        is trimmed to for faster run times.

        This changes the phase settings folder is tagged as follows:

        image_psf_shape = 1 -> settings
        image_psf_shape = 2 -> settings_image_psf_shape_2
        image_psf_shape = 2 -> settings_image_psf_shape_2
        """
        if self.primary_beam_shape_2d is None:
            return ""
        y = str(self.primary_beam_shape_2d[0])
        x = str(self.primary_beam_shape_2d[1])
        return (
            "__"
            + conf.instance.tag.get("phase", "primary_beam_shape", str)
            + "_"
            + y
            + "x"
            + x
        )
