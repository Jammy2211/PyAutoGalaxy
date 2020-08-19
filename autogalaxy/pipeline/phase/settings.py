from autoconf import conf
from autofit.tools.phase import AbstractSettingsPhase
from autogalaxy.dataset import imaging, interferometer
from autoarray.inversion import pixelizations as pix, inversions as inv


class SettingsPhase(AbstractSettingsPhase):
    def __init__(
        self,
        settings_masked_dataset,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        log_likelihood_cap=None,
    ):
        """The settings of a phase, which customize how a model is fitted to data in a PyAutoGalaxy *Phase*. for
        example the type of grid used or options or augmenting the data.

        The *SettingsPhase* perform phase tagging, whereby the phase settings tag the output path of the results
        depending on their parameters This allows one to fit models to a dataset using different settings in a
        structured path format.

        Parameters
        ----------

        """
        super().__init__(log_likelihood_cap=log_likelihood_cap)

        self.settings_masked_dataset = settings_masked_dataset
        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion


class SettingsPhaseImaging(SettingsPhase):
    def __init__(
        self,
        settings_masked_imaging=imaging.SettingsMaskedImaging(),
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        log_likelihood_cap=None,
    ):

        super().__init__(
            settings_masked_dataset=settings_masked_imaging,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            log_likelihood_cap=log_likelihood_cap,
        )

    @property
    def settings_masked_imaging(self):
        return self.settings_masked_dataset

    @property
    def phase_tag_no_inversion(self):
        return (
            conf.instance.tag.get("phase", "phase")
            + self.settings_masked_imaging.tag_no_inversion
            + self.log_likelihood_cap_tag
        )

    @property
    def phase_tag_with_inversion(self):
        return (
            conf.instance.tag.get("phase", "phase")
            + self.settings_masked_imaging.tag_with_inversion
            + self.settings_pixelization.tag
            + self.settings_inversion.tag
            + self.log_likelihood_cap_tag
        )


class SettingsPhaseInterferometer(SettingsPhase):
    def __init__(
        self,
        masked_interferometer=interferometer.SettingsMaskedInterferometer(),
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        log_likelihood_cap=None,
    ):

        super().__init__(
            settings_masked_dataset=masked_interferometer,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            log_likelihood_cap=log_likelihood_cap,
        )

    @property
    def settings_masked_interferometer(self):
        return self.settings_masked_dataset

    @property
    def phase_tag_no_inversion(self):

        return (
            conf.instance.tag.get("phase", "phase")
            + self.settings_masked_interferometer.tag_no_inversion
            + self.log_likelihood_cap_tag
        )

    @property
    def phase_tag_with_inversion(self):

        return (
            conf.instance.tag.get("phase", "phase")
            + self.settings_masked_interferometer.tag_with_inversion
            + self.settings_pixelization.tag
            + self.settings_inversion.tag
            + self.log_likelihood_cap_tag
        )
