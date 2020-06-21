from autoconf import conf
from autofit.tools.phase import AbstractPhaseSettings
from autoarray.structures import grids
from autoarray.operators import transformer


class PhaseSettings(AbstractPhaseSettings):
    def __init__(
        self,
        grid_class=grids.Grid,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        log_likelihood_cap=None,
    ):
        """The settings of a phase, which customize how a model is fitted to data in a PyAutoGalaxy *Phase*. for
        example the type of grid used or options or augmenting the data.

        The *PhaseSettings* perform phase tagging, whereby the phase settings tag the output path of the results
        depending on their parameters This allows one to fit models to a dataset using different settings in a
        structured path format.

        Parameters
        ----------
        grid_class : ag.Grid
            The type of grid used to create the image from the *Galaxy* and *Plane*. The options are *Grid*,
            *GridIterate* and *GridInterpolate* (see the *Grids* documentation for a description of these options).
        grid_inversion_class : ag.Grid
            The type of grid used to create the grid that maps the *Inversion* source pixels to the data's image-pixels.
            The options are *Grid*, *GridIterate* and *GridInterpolate* (see the *Grids* documentation for a
            description of these options).
        sub_size : int
            If the grid and / or grid_inversion use a *Grid*, this sets the sub-size used by the *Grid*.
        fractional_accuracy : float
            If the grid and / or grid_inversion use a *GridIterate*, this sets the fractional accuracy it
            uses when evaluating functions.
        sub_steps : [int]
            If the grid and / or grid_inversion use a *GridIterate*, this sets the steps the sub-size is increased by
            to meet the fractional accuracy when evaluating functions.
        pixel_scales_interp : float or (float, float)
            If the grid and / or grid_inversion use a *GridInterpolate*, this sets the resolution of the interpolation
            grid.
        signal_to_noise_limit : float
            If input, the dataset's noise-map is rescaled such that no pixel has a signal-to-noise above the
            signa to noise limit.
        """
        super().__init__(log_likelihood_cap=log_likelihood_cap)

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
            + conf.instance.tag.get("phase", "grid")
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
            + conf.instance.tag.get("phase", "grid")
            + "_"
            + self.grid_sub_size_tag
            + self.grid_fractional_accuracy_tag
            + self.grid_pixel_scales_interp_tag
            + "_"
            + conf.instance.tag.get("phase", "grid_inversion")
            + "_"
            + self.grid_inversion_sub_size_tag
            + self.grid_inversion_fractional_accuracy_tag
            + self.grid_inversion_pixel_scales_interp_tag
        )

    @property
    def grid_sub_size_tag(self):
        """Generate a sub-size tag, to customize phase names based on the sub-grid size used, of the Grid class.

        This changes the phase settings folder as follows:

        sub_size = None -> settings
        sub_size = 1 -> settings_sub_size_2
        sub_size = 4 -> settings_sub_size_4
        """
        if not self.grid_class is grids.Grid:
            return ""
        return conf.instance.tag.get("phase", "sub_size") + "_" + str(self.sub_size)

    @property
    def grid_fractional_accuracy_tag(self):
        """Generate a fractional accuracy tag, to customize phase names based on the fractional accuracy of the
        GridIterate class.

        This changes the phase settings folder as follows:

        fraction_accuracy = 0.5 -> settings__facc_0.5
        fractional_accuracy = 0.999999 = 4 -> settings__facc_0.999999
        """
        if not self.grid_class is grids.GridIterate:
            return ""
        return (
            conf.instance.tag.get("phase", "fractional_accuracy")
            + "_"
            + str(self.fractional_accuracy)
        )

    @property
    def grid_pixel_scales_interp_tag(self):
        """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the 
        GridInterpolate.

        This changes the phase settings folder as follows:

        pixel_scales_interp = None -> settings
        pixel_scales_interp = 0.1 -> settings___grid_interp_0.1
        """
        if not self.grid_class is grids.GridInterpolate:
            return ""
        if self.pixel_scales_interp is None:
            return ""
        return conf.instance.tag.get(
            "phase", "pixel_scales_interp"
        ) + "_{0:.3f}".format(self.pixel_scales_interp)

    @property
    def grid_inversion_sub_size_tag(self):
        """Generate a sub-size tag, to customize phase names based on the sub-grid size used, of the Grid class.

        This changes the phase settings folder as follows:

        sub_size = None -> settings
        sub_size = 1 -> settings__grid_sub_size_2
        sub_size = 4 -> settings__grid_inv_sub_size_4
        """
        if not self.grid_inversion_class is grids.Grid:
            return ""
        return conf.instance.tag.get("phase", "sub_size") + "_" + str(self.sub_size)

    @property
    def grid_inversion_fractional_accuracy_tag(self):
        """Generate a fractional accuracy tag, to customize phase names based on the fractional accuracy of the
        GridIterate class.

        This changes the phase settings folder as follows:

        fraction_accuracy = 0.5 -> settings__facc_0.5
        fractional_accuracy = 0.999999 = 4 -> settings__facc_0.999999
        """
        if not self.grid_inversion_class is grids.GridIterate:
            return ""
        return (
            conf.instance.tag.get("phase", "fractional_accuracy")
            + "_"
            + str(self.fractional_accuracy)
        )

    @property
    def grid_inversion_pixel_scales_interp_tag(self):
        """Generate an interpolation pixel scale tag, to customize phase names based on the resolution of the 
        GridInterpolate.

        This changes the phase settings folder as follows:

        pixel_scales_interp = None -> settings
        pixel_scales_interp = 0.1 -> settings___grid_interp_0.1
        """
        if not self.grid_inversion_class is grids.GridInterpolate:
            return ""
        if self.pixel_scales_interp is None:
            return ""
        return conf.instance.tag.get(
            "phase", "pixel_scales_interp"
        ) + "_{0:.3f}".format(self.pixel_scales_interp)

    @property
    def signal_to_noise_limit_tag(self):
        """Generate a signal to noise limit tag, to customize phase names based on limiting the signal to noise ratio of
        the dataset being fitted.

        This changes the phase settings folder as follows:

        signal_to_noise_limit = None -> settings
        signal_to_noise_limit = 2 -> settings_snr_2
        signal_to_noise_limit = 10 -> settings_snr_10
        """
        if self.signal_to_noise_limit is None:
            return ""
        return (
            "__"
            + conf.instance.tag.get("phase", "signal_to_noise_limit")
            + "_"
            + str(self.signal_to_noise_limit)
        )

    @property
    def bin_up_factor_tag(self):
        """Generate a bin up tag, to customize phase names based on the resolutioon the image is binned up by for faster \
        run times.

        This changes the phase settings folder as follows:

        bin_up_factor = 1 -> settings
        bin_up_factor = 2 -> settings_bin_up_factor_2
        bin_up_factor = 2 -> settings_bin_up_factor_2
        """
        if self.bin_up_factor == 1 or self.bin_up_factor is None:
            return ""
        return (
            "__"
            + conf.instance.tag.get("phase", "bin_up_factor")
            + "_"
            + str(self.bin_up_factor)
        )


class PhaseSettingsImaging(PhaseSettings):
    def __init__(
        self,
        grid_class=grids.Grid,
        grid_inversion_class=grids.Grid,
        sub_size=2,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
        signal_to_noise_limit=None,
        bin_up_factor=None,
        inversion_pixel_limit=None,
        psf_shape_2d=None,
        log_likelihood_cap=None,
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
            log_likelihood_cap=log_likelihood_cap,
        )

        self.psf_shape_2d = psf_shape_2d

    @property
    def phase_no_inversion_tag(self):
        return (
            conf.instance.tag.get("phase", "phase")
            + self.grid_no_inversion_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.psf_shape_tag
            + self.log_likelihood_cap_tag
        )

    @property
    def phase_with_inversion_tag(self):
        return (
            conf.instance.tag.get("phase", "phase")
            + self.grid_with_inversion_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.psf_shape_tag
            + self.log_likelihood_cap_tag
        )

    @property
    def psf_shape_tag(self):
        """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
        is trimmed to for faster run times.

        This changes the phase settings folder as follows:

        image_psf_shape = 1 -> settings
        image_psf_shape = 2 -> settings_image_psf_shape_2
        image_psf_shape = 2 -> settings_image_psf_shape_2
        """
        if self.psf_shape_2d is None:
            return ""
        y = str(self.psf_shape_2d[0])
        x = str(self.psf_shape_2d[1])
        return "__" + conf.instance.tag.get("phase", "psf_shape") + "_" + y + "x" + x


class PhaseSettingsInterferometer(PhaseSettings):
    def __init__(
        self,
        grid_class=grids.Grid,
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
        log_likelihood_cap=None,
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
            log_likelihood_cap=log_likelihood_cap,
        )

        self.transformer_class = transformer_class
        self.primary_beam_shape_2d = primary_beam_shape_2d

    @property
    def phase_no_inversion_tag(self):

        return (
            conf.instance.tag.get("phase", "phase")
            + self.grid_no_inversion_tag
            + self.transformer_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.primary_beam_shape_tag
            + self.log_likelihood_cap_tag
        )

    @property
    def phase_with_inversion_tag(self):

        return (
            conf.instance.tag.get("phase", "phase")
            + self.grid_with_inversion_tag
            + self.transformer_tag
            + self.signal_to_noise_limit_tag
            + self.bin_up_factor_tag
            + self.primary_beam_shape_tag
            + self.log_likelihood_cap_tag
        )

    @property
    def transformer_tag(self):
        """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
        is trimmed to for faster run times.

        This changes the phase settings folder as follows:

        image_psf_shape = 1 -> settings
        image_psf_shape = 2 -> settings_image_psf_shape_2
        image_psf_shape = 2 -> settings_image_psf_shape_2
        """
        if self.transformer_class is None:
            return ""
        return "__" + conf.instance.tag.get(
            "transformer", self.transformer_class.__name__
        )

    @property
    def primary_beam_shape_tag(self):
        """Generate an image psf shape tag, to customize phase names based on size of the image PSF that the original PSF \
        is trimmed to for faster run times.

        This changes the phase settings folder as follows:

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
            + conf.instance.tag.get("phase", "primary_beam_shape")
            + "_"
            + y
            + "x"
            + x
        )
