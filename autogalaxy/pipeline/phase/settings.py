from autoconf import conf
from autofit.tools.phase import AbstractPhaseSettings
from autogalaxy.dataset import imaging, interferometer
from autoarray.inversion import pixelizations as pix, inversions as inv
from autoarray.structures import grids
from autoarray.operators import transformer


class PhaseSettings(AbstractPhaseSettings):
    def __init__(
        self,
        masked_dataset_settings,
        pixelization_settings=pix.PixelizationSettings(),
        inversion_settings=inv.InversionSettings(),
        log_likelihood_cap=None,
    ):
        """The settings of a phase, which customize how a model is fitted to data in a PyAutoGalaxy *Phase*. for
        example the type of grid used or options or augmenting the data.

        The *PhaseSettings* perform phase tagging, whereby the phase settings tag the output path of the results
        depending on their parameters This allows one to fit models to a dataset using different settings in a
        structured path format.

        Parameters
        ----------

        """
        super().__init__(log_likelihood_cap=log_likelihood_cap)

        self.masked_dataset = masked_dataset_settings
        self.pixelization = pixelization_settings
        self.inversion = inversion_settings


class PhaseSettingsImaging(PhaseSettings):
    def __init__(
        self,
        masked_imaging_settings=imaging.MaskedImagingSettings(),
        pixelization_settings=pix.PixelizationSettings(),
        inversion_settings=inv.InversionSettings(),
        log_likelihood_cap=None,
    ):

        super().__init__(
            masked_dataset_settings=masked_imaging_settings,
            pixelization_settings=pixelization_settings,
            inversion_settings=inversion_settings,
            log_likelihood_cap=log_likelihood_cap,
        )

    @property
    def masked_imaging(self):
        return self.masked_dataset

    @property
    def phase_no_inversion_tag(self):
        return (
            conf.instance.tag.get("phase", "phase")
            + self.masked_imaging.grid_no_inversion_tag
            + self.masked_imaging.signal_to_noise_limit_tag
            + self.masked_imaging.bin_up_factor_tag
            + self.masked_imaging.psf_shape_tag
            + self.log_likelihood_cap_tag
        )

    @property
    def phase_with_inversion_tag(self):
        return (
            conf.instance.tag.get("phase", "phase")
            + self.masked_imaging.grid_with_inversion_tag
            + self.masked_imaging.signal_to_noise_limit_tag
            + self.masked_imaging.bin_up_factor_tag
            + self.masked_imaging.psf_shape_tag
            + self.log_likelihood_cap_tag
        )


class PhaseSettingsInterferometer(PhaseSettings):
    def __init__(
        self,
        masked_interferometer_settings=interferometer.MaskedInterferometerSettings(),
        pixelization_settings=pix.PixelizationSettings(),
        inversion_settings=inv.InversionSettings(),
        log_likelihood_cap=None,
    ):

        super().__init__(
            masked_dataset_settings=masked_interferometer_settings,
            pixelization_settings=pixelization_settings,
            inversion_settings=inversion_settings,
            log_likelihood_cap=log_likelihood_cap,
        )

    @property
    def masked_interferometer(self):
        return self.masked_dataset

    @property
    def phase_no_inversion_tag(self):

        return (
            conf.instance.tag.get("phase", "phase")
            + self.masked_interferometer.grid_no_inversion_tag
            + self.masked_interferometer.transformer_tag
            + self.masked_interferometer.signal_to_noise_limit_tag
            + self.log_likelihood_cap_tag
        )

    @property
    def phase_with_inversion_tag(self):

        return (
            conf.instance.tag.get("phase", "phase")
            + self.masked_interferometer.grid_with_inversion_tag
            + self.masked_interferometer.transformer_tag
            + self.inversion.use_linear_operators_tag
            + self.masked_interferometer.signal_to_noise_limit_tag
            + self.log_likelihood_cap_tag
        )
