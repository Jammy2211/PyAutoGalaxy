from autoconf import conf
import autofit as af

from autoarray.inversion import pixelizations as pix, regularization as reg
from autogalaxy.profiles import (
    light_profiles as lp,
    mass_profiles as mp,
    light_and_mass_profiles as lmp,
)
from autogalaxy.hyper import hyper_data as hd
from autogalaxy import exc

from typing import Union, Optional

import copy


class AbstractSetup:
    def _cls_to_prior_model(self, cls):
        """
        Convert an input class to a `af.PriorModel` so that if a user specifies the models input into a `Setup` not as
        a `PriorModel` (or uses the default inputs which are not `PriorModel`'s) they are converted to a `PriorModel`
        for the pipeline.

        If `None` or a `PriorModel` is input it is not converted to a `PriorModel`.

        Parameters
        ----------
        cls : cls
            The class which is converted to a `PriorModel`' if it is not a `PriorModel`.

        Returns
        -------
        af.PriorModel or None
            The converted class.

        """

        if cls is not None:
            if not isinstance(cls, af.PriorModel):
                return af.PriorModel(cls)

        return cls

    def set_light_prior_model_assertions(
        self,
        bulge_prior_model=None,
        disk_prior_model=None,
        envelope_prior_model=None,
        assert_bulge_sersic_above_disk=True,
        assert_chameleon_core_radius_0_above_core_radius_1=True,
    ):
        """
        This sets a number of assertions on the bugle, disk and envelope prior models which can be customized via
        boolean operators. These assertions are:
        
        1) That the bulge `sersic_index` is above the disk `sersic_index`, if both components are Sersic profiles.
        2) That `core_radius_0` of every Chameleon profile is less than `core_radius_1`.
        
        Parameters
        ----------
        bulge_prior_model : af.PriorModel
            The `PriorModel` used to represent the light distribution of a bulge.
        disk_prior_model : af.PriorModel
            The `PriorModel` used to represent the light distribution of a disk.
        envelope_prior_model : af.PriorModel
            The `PriorModel` used to represent the light distribution of a envelope.
        assert_bulge_sersic_above_disk : bool
            If `True`, the `sersic_index` of the bulge is above that of the disk.
        assert_chameleon_core_radius_0_above_core_radius_1 : bool
            If `True`, `core_radius_0` of every Chameleon profile is above its `core_radius_1` value.

        Returns
        -------
        None
        """

        if assert_bulge_sersic_above_disk:

            self.set_bulge_disk_assertions(
                bulge_prior_model=bulge_prior_model, disk_prior_model=disk_prior_model
            )

        if assert_chameleon_core_radius_0_above_core_radius_1:

            self.set_chameleon_assertions(prior_model=bulge_prior_model)
            self.set_chameleon_assertions(prior_model=disk_prior_model)
            self.set_chameleon_assertions(prior_model=envelope_prior_model)

    def set_bulge_disk_assertions(self, bulge_prior_model, disk_prior_model):
        """
        Sets an assertion on the `bulge_prior_model` and `disk_prior_model` such that the `sersic_index` of the
        bulge is higher than that of the `disk`, if both components are modeled using Sersic profiles.

        Parameters
        ----------
        bulge_prior_model : af.PriorModel
            The `PriorModel` used to represent the light distribution of a bulge.
        disk_prior_model : af.PriorModel
            The `PriorModel` used to represent the light distribution of a disk.

        Returns
        -------
        None
        """

        if bulge_prior_model is not None and disk_prior_model is not None:
            if hasattr(bulge_prior_model, "sersic_index") and hasattr(
                disk_prior_model, "sersic_index"
            ):
                bulge_prior_model.add_assertion(
                    bulge_prior_model.sersic_index > disk_prior_model.sersic_index
                )

    def set_chameleon_assertions(self, prior_model):
        """
        Sets the assertion on all `PriorModels` which are a `Chameleon` profile such that the core radius of the first
        isothermal profile is lower than the second, preventing negative mass.

        Parameters
        ----------
        prior_model : af.PriorModel
            The `PriorModel` that may contain a `Chameleon` profile.

        Returns
        -------
        None
        """
        if prior_model is not None:
            if hasattr(prior_model, "core_radius_0") and hasattr(
                prior_model, "core_radius_1"
            ):
                prior_model.add_assertion(
                    prior_model.core_radius_0 < prior_model.core_radius_1
                )


class SetupHyper(AbstractSetup):
    def __init__(
        self,
        hyper_galaxies: bool = False,
        hyper_image_sky: Optional[type(hd.HyperImageSky)] = None,
        hyper_background_noise: Optional[type(hd.HyperBackgroundNoise)] = None,
        search: Optional[af.NonLinearSearch] = None,
        evidence_tolerance: Optional[float] = None,
    ):
        """
        The hyper setup of a pipeline, which controls how hyper-features in PyAutoGalaxy template pipelines run,
        for example controlling whether hyper galaxies are used to scale the noise and the non-linear searches used
        in these phases.

        Users can write their own pipelines which do not use or require the *SetupHyper* class.

        Parameters
        ----------
        hyper_galaxies : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used to scale the
            noise-map of the dataset throughout the fitting.
        hyper_image_sky : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            image's background sky component in the model.
        hyper_background_noise : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            noise-map's background component in the model.
        hyper_search_no_inversion : af.NonLinearSearch or None
            The `NonLinearSearch` used by every inversion phase.
        hyper_search_with_inversion : af.NonLinearSearch or None
            The `NonLinearSearch` used by every hyper combined phase.
        evidence_tolerance : float
            The evidence tolerance of the non-linear searches used in the hyper phases, whereby higher values will
            lead them to end earlier at the expense of accuracy.
        """

        self.evidence_tolerance = evidence_tolerance

        if evidence_tolerance is not None:
            if search is not None:
                raise exc.PipelineException(
                    "You have manually specified a search in the SetupPipeline, and an evidence_tolerance."
                    "You cannot manually specify both - remove one."
                    "(If you want the hyper search to use a specific evidence tolerance, include the evidence"
                    "tolerance in its parameters"
                )

        self.hyper_galaxies = hyper_galaxies

        self.hyper_galaxy_names = None

        if search is None:
            self.search = af.DynestyStatic(
                n_live_points=50,
                evidence_tolerance=self.evidence_tolerance,
                sample="rstagger",
            )
        elif search is not None:
            self.search = search

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

    @property
    def hypers_all_off(self):
        if not self.hyper_galaxies:
            if self.hyper_image_sky is None and self.hyper_background_noise is None:
                return True
        return False

    @property
    def hypers_all_except_image_sky_off(self):
        if not self.hyper_galaxies:
            if self.hyper_background_noise is None:
                return True
        return False

    def hyper_image_sky_from_result(self, result: af.Result, as_model=True):

        if self.hyper_image_sky is not None:
            if as_model:
                if hasattr(result, "hyper"):
                    return result.hyper.model.hyper_image_sky
                return result.model.hyper_image_sky
            if hasattr(result, "hyper"):
                return result.hyper.instance.hyper_image_sky
            return result.instance.hyper_image_sky

    def hyper_background_noise_from_result(self, result: af.Result):

        if self.hyper_background_noise is not None:
            if hasattr(result, "hyper"):
                return result.hyper.instance.hyper_background_noise
            return result.instance.hyper_background_noise


class AbstractSetupLight(AbstractSetup):

    pass


class SetupLightParametric(AbstractSetupLight):
    def __init__(
        self,
        bulge_prior_model: af.PriorModel(lp.LightProfile) = lp.EllipticalSersic,
        disk_prior_model: af.PriorModel(lp.LightProfile) = lp.EllipticalExponential,
        envelope_prior_model: af.PriorModel(lp.LightProfile) = None,
        light_centre: (float, float) = None,
        light_centre_gaussian_prior_values: (float, float) = None,
        align_bulge_disk_centre: bool = True,
        align_bulge_disk_elliptical_comps: bool = False,
        align_bulge_envelope_centre: bool = False,
        assert_bulge_sersic_above_disk=True,
    ):
        """
        The setup of the parametric light modeling in a pipeline, which controls how PyAutoGalaxy template pipelines
        run, for example controlling assumptions about the bulge-disk model.

        Users can write their own pipelines which do not use or require the `SetupLightParametric` class.

        Parameters
        ----------
        bulge_prior_model : af.PriorModel(lp.LightProfile)
            The `LightProfile` `PriorModel` used to represent the light distribution of a bulge.
        disk_prior_model : af.PriorModel(lp.LightProfile)
            The `LightProfile` `PriorModel` used to represent the light distribution of a disk.
        envelope_prior_model : af.PriorModel(lp.LightProfile)
            The `LightProfile` `PriorModel` used to represent the light distribution of a envelope.
        light_centre : (float, float) or None
           If input, a fixed (y,x) centre of the galaxy is used for the light profile model which is not treated as a
            free parameter by the non-linear search.
        align_bulge_disk_centre : bool or None
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            `True` will align the centre of the bulge and disk components and not fit them separately.
        align_bulge_disk_elliptical_comps : bool or None
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            `True` will align the elliptical components the bulge and disk components and not fit them separately.
        align_bulge_envelope_centre : bool or None
            If a bulge + envelope light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the
            galaxy, `True` will align the centre of the bulge and envelope components and not fit them separately.
        """

        self.bulge_prior_model = self._cls_to_prior_model(cls=bulge_prior_model)
        self.disk_prior_model = self._cls_to_prior_model(cls=disk_prior_model)
        self.envelope_prior_model = self._cls_to_prior_model(cls=envelope_prior_model)

        self.light_centre_gaussian_prior_values = light_centre_gaussian_prior_values

        if self.light_centre_gaussian_prior_values is not None:

            mean = light_centre_gaussian_prior_values[0]
            sigma = light_centre_gaussian_prior_values[1]

            if self.bulge_prior_model is not None:

                self.bulge_prior_model.centre_0 = af.GaussianPrior(
                    mean=mean, sigma=sigma
                )
                self.bulge_prior_model.centre_1 = af.GaussianPrior(
                    mean=mean, sigma=sigma
                )

            if self.disk_prior_model is not None:

                self.disk_prior_model.centre_0 = af.GaussianPrior(
                    mean=mean, sigma=sigma
                )
                self.disk_prior_model.centre_1 = af.GaussianPrior(
                    mean=mean, sigma=sigma
                )

            if self.envelope_prior_model is not None:

                self.envelope_prior_model.centre_0 = af.GaussianPrior(
                    mean=mean, sigma=sigma
                )
                self.envelope_prior_model.centre_1 = af.GaussianPrior(
                    mean=mean, sigma=sigma
                )

        self.light_centre = light_centre
        self.align_bulge_disk_centre = align_bulge_disk_centre
        self.align_bulge_disk_elliptical_comps = align_bulge_disk_elliptical_comps

        if self.bulge_prior_model is not None and self.disk_prior_model is not None:

            if self.align_bulge_disk_centre:

                self.bulge_prior_model.centre = self.disk_prior_model.centre

            if self.align_bulge_disk_elliptical_comps:

                if hasattr(self.bulge_prior_model, "elliptical_comps") and hasattr(
                    self.disk_prior_model, "elliptical_comps"
                ):
                    self.bulge_prior_model.elliptical_comps = (
                        self.disk_prior_model.elliptical_comps
                    )
        self.align_bulge_envelope_centre = align_bulge_envelope_centre

        if self.bulge_prior_model is not None and self.envelope_prior_model is not None:

            if self.align_bulge_envelope_centre:

                self.envelope_prior_model.centre = self.bulge_prior_model.centre

        if self.light_centre is not None:

            if self.bulge_prior_model is not None:

                self.bulge_prior_model.centre = self.light_centre

            if self.disk_prior_model is not None:

                self.disk_prior_model.centre = self.light_centre

            if self.envelope_prior_model is not None:

                self.envelope_prior_model.centre = self.light_centre

        self.set_light_prior_model_assertions(
            bulge_prior_model=self.bulge_prior_model,
            disk_prior_model=self.disk_prior_model,
            envelope_prior_model=self.envelope_prior_model,
            assert_bulge_sersic_above_disk=assert_bulge_sersic_above_disk,
        )


class SetupLightInversion(AbstractSetupLight):
    def __init__(
        self,
        pixelization_prior_model: af.PriorModel(pix.Pixelization),
        regularization_prior_model: af.PriorModel(reg.Regularization),
        inversion_pixels_fixed: float = None,
    ):
        """
        The setup of the inversion light modeling of a pipeline, which controls how PyAutoGalaxy template pipelines run,
        for example controlling the `Pixelization` and `Regularization` used by the `Inversion`.

        Users can write their own pipelines which do not use or require the `SetupLightInversion` class.

        Parameters
        ----------
        pixelization_prior_model : af.PriorModel(pix.Pixelization)
           If the pipeline uses an `Inversion` to reconstruct the galaxy's light, this determines the `Pixelization`
           used.
        regularization_prior_model : af.PriorModel(reg.Regularization)
            If the pipeline uses an `Inversion` to reconstruct the galaxy's light, this determines the `Regularization`
            scheme used.
        inversion_pixels_fixed : float
            The fixed number of source pixels used by a `Pixelization` class that takes as input a fixed number of
            pixels.
        """

        self._pixelization_prior_model = self._cls_to_prior_model(
            cls=pixelization_prior_model
        )
        self.regularization_prior_model = self._cls_to_prior_model(
            cls=regularization_prior_model
        )

        self.inversion_pixels_fixed = inversion_pixels_fixed

    @property
    def pixelization_prior_model(self) -> af.PriorModel:
        """
        The `PriorModel` of the `Pixelization` in the pipeline.

        This `PriorModel` has its number of pixels fixed to a certain value if the `inversion_pixels_fixed` parameter
        is input.
        """
        if (
            self._pixelization_prior_model.cls is not pix.VoronoiBrightnessImage
            or self.inversion_pixels_fixed is None
        ):
            return self._pixelization_prior_model

        pixelization_prior_model = self._pixelization_prior_model
        pixelization_prior_model.pixels = self.inversion_pixels_fixed
        return pixelization_prior_model


class AbstractSetupMass(AbstractSetup):
    def __init__(self, mass_centre: (float, float) = None):
        """
        The setup of the mass modeling in a pipeline for `MassProfile`'s representing the total (e.g. stars + dark
        matter) mass distribution, which controls how PyAutoGalaxy template pipelines run, for example controlling
        assumptions about the bulge-disk model.

        Users can write their own pipelines which do not use or require the `` class.

        Parameters
        ----------
        mass_centre : (float, float)
           If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
           non-linear search.
        """

        self.mass_centre = mass_centre

    def align_centre_to_mass_centre(
        self, mass_prior_model: af.PriorModel(mp.MassProfile)
    ) -> af.PriorModel:
        """
        Align the centre of an input `MassProfile` `PriorModel` to the `mass_centre` of this pipeline setup, such
        that in the model the centre of the mass profile is fixed and not a free parameters that is fitted for.

        If the `mass_centre` is None the mass profile centre is unchanged and remains a model.

        Parameters
        ----------
        mass_prior_model : af.PriorModel(ag.mp.MassProfile)
            The `MassProfile` whose centre may be aligned with the mass_centre attribute.
        """
        if self.mass_centre is not None:
            mass_prior_model.centre = self.mass_centre
        return mass_prior_model

    def unfix_mass_centre(
        self, result: af.Result, mass_prior_model: af.PriorModel(mp.MassProfile)
    ) -> af.PriorModel:
        """
        If the centre of the mass `PriorModel` was previously fixed to an input value via the `mass_centre` input,
        unalign them by making their centre `GaussianPrior`'s with `mean` centred on the input `mass_centre`.

        If `mass_centre` was not input an the centre was fixed in the pipeline itsef, this function can be used to
        unfix the centre to the model result of a previous phase in the pipeline.

        Parameters
        ----------
        result : af.Result
            The result of the previous source or light pipeline.        
        mass_prior_model : af.PriorModel(ag.mp.MassProfile)
            The mass profile whose centre may be unfixed from a previous model.
        """

        if self.mass_centre is not None:

            mass_prior_model.centre.centre_0 = af.GaussianPrior(
                mean=self.mass_centre[0], sigma=0.05
            )
            mass_prior_model.centre.centre_1 = af.GaussianPrior(
                mean=self.mass_centre[1], sigma=0.05
            )

        else:

            mass_prior_model.centre = result.last.model.galaxies.lens.mass.centre

        return mass_prior_model


class SetupMassTotal(AbstractSetupMass):
    def __init__(
        self,
        mass_prior_model: af.PriorModel(mp.MassProfile) = mp.EllipticalPowerLaw,
        mass_centre: (float, float) = None,
        align_bulge_mass_centre: bool = False,
    ):
        """
        The setup of the mass modeling in a pipeline for `MassProfile`'s representing the total (e.g. stars + dark
        matter) mass distribution, which controls how PyAutoGalaxy template pipelines run, for example controlling
        assumptions about the bulge-disk model.

        Users can write their own pipelines which do not use or require the `SetupMassTotal` class.

        Parameters
        ----------
        mass_prior_model : af.PriorModel(mp.MassProfile)
            The `MassProfile` fitted by the `Pipeline` (the pipeline must specifically use this option to use this
            mass profile)
        mass_centre : (float, float) or None
           If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
           non-linear search.
        align_bulge_mass_centre : bool
            If `True` and the galaxy model has both a light and mass component, the function
            `align_centre_of_mass_to_light` can be used to align their centres.
        """

        super().__init__(mass_centre=mass_centre)

        self.mass_prior_model = self._cls_to_prior_model(cls=mass_prior_model)

        self.align_bulge_mass_centre = align_bulge_mass_centre

        if self.mass_centre is not None:
            self.mass_prior_model.centre = self.mass_centre

    def align_centre_of_mass_to_light(
        self,
        mass_prior_model: af.PriorModel(mp.MassProfile),
        light_centre: (float, float),
    ):
        """Align the centre of a mass profile to the centre of a light profile, if the align_bulge_mass_centre
        SLaM setting is True.

        Parameters
        ----------
        mass_prior_model : af.PriorModel(ag.mp.MassProfile)
            The mass profile whose centre may be aligned with the light_centre attribute.
        light : (float, float)
            The centre of the light profile the mass profile is aligned with.
        """
        if self.align_bulge_mass_centre:
            mass_prior_model.centre = light_centre
        else:
            mass_prior_model.centre.centre_0 = af.GaussianPrior(
                mean=light_centre[0], sigma=0.1
            )
            mass_prior_model.centre.centre_1 = af.GaussianPrior(
                mean=light_centre[1], sigma=0.1
            )
        return mass_prior_model

    def unalign_mass_centre_from_light_centre(
        self,
        results: af.ResultsCollection,
        mass_prior_model: af.PriorModel(mp.MassProfile),
    ):
        """If the centre of a mass model was previously aligned with that of the lens light centre, unaligned them
        by using an earlier model of the light.

        Parameters
        ----------
        mass_prior_model : af.PriorModel(ag.mp.MassProfile)
            The `MassProfile` whose centre may be aligned with the `LightProfile` centre.
        """
        if self.align_bulge_mass_centre:

            mass_prior_model.centre = results[-3].model.galaxies.lens.bulge.centre

        else:

            mass_prior_model.centre = results.last.model.galaxies.lens.mass.centre

        return mass_prior_model

    def mass_prior_model_with_updated_priors_from_result(
        self, result: af.Result, unfix_mass_centre=False
    ):
        """
        Returns an updated version of the `mass_prior_model` whose priors are initialized from previous results in a
        pipeline.

        This function generically links any `MassProfile` to any `MassProfile`, pairing parameters which share the
        same path.

        Parameters
        ----------
        results : af.ResultsCollection
            The results of the previous mass pipeline.

        Returns
        -------
        af.PriorModel(mp.MassProfile)
            The total mass profile whose priors are initialized from a previous result.
        """

        mass = self._cls_to_prior_model(cls=self.mass_prior_model.cls)

        mass.take_attributes(source=result.model.galaxies.lens.mass)

        if unfix_mass_centre and isinstance(mass.centre, tuple):

            centre_tuple = mass.centre

            mass.centre = self._cls_to_prior_model(cls=self.mass_prior_model.cls).centre

            mass.centre.centre_0 = af.GaussianPrior(mean=centre_tuple[0], sigma=0.05)
            mass.centre.centre_1 = af.GaussianPrior(mean=centre_tuple[1], sigma=0.05)

        return mass


class SetupMassLightDark(AbstractSetupMass):
    def __init__(
        self,
        bulge_prior_model: af.PriorModel(lmp.LightMassProfile) = lmp.EllipticalSersic,
        disk_prior_model: af.PriorModel(
            lmp.LightMassProfile
        ) = lmp.EllipticalExponential,
        envelope_prior_model: af.PriorModel(lmp.LightMassProfile) = None,
        dark_prior_model: af.PriorModel(mp.MassProfile) = mp.EllipticalNFWMCRLudlow,
        mass_centre: (float, float) = None,
        constant_mass_to_light_ratio: bool = False,
        align_bulge_dark_centre: bool = False,
    ):
        """
        The setup of the mass modeling in a pipeline for `MassProfile`'s representing the decomposed light and dark
        mass distributions, which controls how PyAutoGalaxy template pipelines run, for example controlling assumptions
        about the bulge-disk model.

        Users can write their own pipelines which do not use or require the `SetupMassLightDark` class.

        Parameters
        ----------
        bulge_prior_model : af.PriorModel(lmp.LightMassProfile)
            The `LightMassProfile` `PriorModel` used to represent the light and mass distribution of the bulge.
        disk_prior_model : af.PriorModel(lmp.LightMassProfile)
            The `LightMassProfile` `PriorModel` used to represent the light and mass distribution of the disk.
        envelope_prior_model : af.PriorModel(lmp.LightMassProfile)
            The `LightMassProfile` `PriorModel` used to represent the light and mass distribution of the stellar envelope.
        dark_prior_model : af.PriorModel(mp.MassProfile)
            The `MassProfile` `PriorModel` used to represent the dark matter distribution of the dark matter halo.
        mass_centre : (float, float)
           If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
           non-linear search.
        constant_mass_to_light_ratio : bool
            If True, and the mass model consists of multiple `LightProfile` and `MassProfile` coomponents, the
            mass-to-light ratio's of all components are fixed to one shared value.
        align_bulge_mass_centre : bool
            If True, and the mass model is a decomposed bulge, disk and dark matter model (e.g. EllipticalSersic +
            EllipticalExponential + SphericalNFW), the centre of the bulge and dark matter profiles are aligned.
        """

        super().__init__(mass_centre=mass_centre)

        self.bulge_prior_model = self._cls_to_prior_model(cls=bulge_prior_model)
        self.disk_prior_model = self._cls_to_prior_model(cls=disk_prior_model)
        self.envelope_prior_model = self._cls_to_prior_model(cls=envelope_prior_model)

        self.dark_prior_model = self._cls_to_prior_model(cls=dark_prior_model)

        self.constant_mass_to_light_ratio = constant_mass_to_light_ratio
        self.align_bulge_dark_centre = align_bulge_dark_centre

        if self.align_bulge_dark_centre:
            self.dark_prior_model.centre = self.bulge_prior_model.centre

        if self.constant_mass_to_light_ratio:
            for profile in self.light_and_mass_prior_models[1:]:
                profile.mass_to_light_ratio = self.light_and_mass_prior_models[
                    0
                ].mass_to_light_ratio

    @property
    def light_and_mass_prior_models(self):
        return list(
            filter(
                None,
                [
                    self.bulge_prior_model,
                    self.disk_prior_model,
                    self.envelope_prior_model,
                ],
            )
        )

    def align_bulge_and_dark_centre(self, bulge_prior_model, dark_prior_model):
        """
        Align the centre of input bulge `PriorModel` with that of the `PriorModel` representing the dark `MassProfile`,
        depending on the `align_bulge_darl_centre` attribute of the `SetupMassLightDark` instance.

        Parameters
        ----------
        bulge_prior_model : af.PriorModel(ag.lmp.LightMassProfile)
            The `LightMassProfile` representing the bulge whose centre is aligned with that of the dark matter.
        disk_prior_model : af.PriorModel(ag.lmp.LightMassProfile)
            The `LightMassProfile` representing the dark matter whose centre is aligned with that of the bulge.
        """
        if self.align_bulge_dark_centre:
            dark_prior_model.centre = bulge_prior_model.centre

    def light_and_mass_prior_models_with_updated_priors(
        self, result: af.Result, einstein_mass_range=None, as_instance=False
    ):
        """
        Returns an updated version of the `bulge_prior_model`, `disk_prior_model`_ and `envelope_prior_model`,  whose
        priors are initialized from previous results of a `Light` pipeline.

        This function generically links any `LightProfile` to any `LightProfile`, pairing parameters which share the
        same path

        Parameters
        ----------
        result : af.ResultsCollection
            The results of the previous source pipeline.

        Returns
        -------
        af.PriorModel(mp.MassProfile)
            The light profiles whose priors are initialized from a previous result.
        """

        if self.bulge_prior_model is None:
            bulge = None
        else:
            bulge = copy.deepcopy(self.bulge_prior_model)

            if as_instance:
                bulge.take_attributes(source=result.instance.galaxies.lens.bulge)
            else:
                bulge.take_attributes(source=result.model.galaxies.lens.bulge)

            if einstein_mass_range is not None:

                bulge = self.update_stellar_mass_priors_from_result(
                    prior_model=bulge,
                    result=result,
                    einstein_mass_range=einstein_mass_range,
                )

        if self.disk_prior_model is None:
            disk = None
        else:
            disk = copy.deepcopy(self.disk_prior_model)

            if as_instance:
                disk.take_attributes(source=result.instance.galaxies.lens.disk)
            else:
                disk.take_attributes(source=result.model.galaxies.lens.disk)

            if einstein_mass_range is not None:
                disk = self.update_stellar_mass_priors_from_result(
                    prior_model=disk,
                    result=result,
                    einstein_mass_range=einstein_mass_range,
                )

        if self.envelope_prior_model is None:
            envelope = None
        else:
            envelope = copy.deepcopy(self.envelope_prior_model)

            if as_instance:
                envelope.take_attributes(source=result.instance.galaxies.lens.envelope)
            else:
                envelope.take_attributes(source=result.model.galaxies.lens.envelope)

            if einstein_mass_range is not None:
                envelope = self.update_stellar_mass_priors_from_result(
                    prior_model=envelope,
                    result=result,
                    einstein_mass_range=einstein_mass_range,
                )

        ### TODO : Assertiosn must be after take attributwes, hence this.

        self.set_light_prior_model_assertions(
            bulge_prior_model=bulge,
            disk_prior_model=disk,
            envelope_prior_model=envelope,
        )

        return bulge, disk, envelope

    def update_stellar_mass_priors_from_result(
        self, prior_model, result: af.Result, einstein_mass_range, bins=100
    ):

        if prior_model is None:
            return None

        grid = result.max_log_likelihood_fit.grid

        einstein_radius = result.max_log_likelihood_tracer.einstein_radius_from_grid(
            grid=grid
        )

        einstein_mass = result.max_log_likelihood_tracer.einstein_mass_angular_from_grid(
            grid=grid
        )

        einstein_mass_lower = einstein_mass_range[0] * einstein_mass
        einstein_mass_upper = einstein_mass_range[1] * einstein_mass

        instance = prior_model.instance_from_prior_medians()

        mass_to_light_ratio_lower = instance.normalization_from_mass_angular_and_radius(
            mass_angular=einstein_mass_lower, radius=einstein_radius, bins=bins
        )
        mass_to_light_ratio_upper = instance.normalization_from_mass_angular_and_radius(
            mass_angular=einstein_mass_upper, radius=einstein_radius, bins=bins
        )

        prior_model.mass_to_light_ratio = af.LogUniformPrior(
            lower_limit=mass_to_light_ratio_lower, upper_limit=mass_to_light_ratio_upper
        )

        return prior_model


class SetupSMBH(AbstractSetup):
    def __init__(
        self,
        smbh_prior_model: af.PriorModel(mp.MassProfile) = mp.PointMass,
        smbh_centre_fixed: bool = True,
    ):
        """
        The setup of a super massive black hole (SMBH) in the mass model of a PyAutoGalaxy template pipeline.

        Users can write their own pipelines which do not use or require the *SetupSMBH* class.

        Parameters
        ----------
        smbh_prior_model : af.PriorModel(mp.MassProfile)
            The `MassProfile` used to model the mass of the super massive black hole.
        smbh_centre_fixed : bool
            If True, the super-massive black hole's centre is fixed to a value input by the pipeline, else it is
            free to vary in the model.
        """
        self.smbh_prior_model = self._cls_to_prior_model(cls=smbh_prior_model)
        self.smbh_centre_fixed = smbh_centre_fixed

    def smbh_from_centre(self, centre, centre_sigma=0.1) -> af.PriorModel:
        """
        Returns a `PriorModel` of the `smbh_prior_model` which is fitted for in the mass-model representing a
        super-massive black-hole (smbh).

        The centre of the smbh is an input parameter of the function, and this centre is either fixed to the input
        values as an instance or fitted for as a model.

        Parameters
        ----------
        centre : (float, float)
            The centre of the `PointMass` that repreents the super-massive black hole.
        centre_sigma : float
            If the centre is free, this is the sigma value of each centre's _GaussianPrior_.
        """

        smbh = af.PriorModel(mp.PointMass)

        if self.smbh_centre_fixed:
            smbh.centre = centre
        else:
            smbh.centre.centre_0 = af.GaussianPrior(mean=centre[0], sigma=centre_sigma)
            smbh.centre.centre_1 = af.GaussianPrior(mean=centre[1], sigma=centre_sigma)

        return smbh


class SetupPipeline:
    def __init__(
        self,
        path_prefix: str = None,
        redshift_galaxy: float = 1.0,
        setup_hyper: SetupHyper = None,
        setup_light: Union[SetupLightParametric, SetupLightInversion] = None,
        setup_mass: Union[SetupMassTotal, SetupMassLightDark] = None,
        setup_smbh: SetupSMBH = None,
    ):
        """
        The setup of a pipeline, which controls how PyAutoGalaxy template pipelines runs, for example controlling
        assumptions about the light and mass models.

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

        Parameters
        ----------
        path_prefix : str or None
            The prefix of folders between the output path and the search folders.
        redshift_galaxy : float
            The redshift of the galaxy fitted, used by the pipeline for converting arc-seconds to kpc, masses to
            solMass, etc.
        setup_hyper : SetupHyper
            The setup of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
        setup_light : SetupLightParametric or SetupLightInversion
            The setup of the light profile modeling (e.g. for bulge-disk models if they are geometrically aligned).
        setup_mass : SetupMassTotal or SetupMassLighDark
            The setup of the mass modeling (e.g. if a constant mass to light ratio is used).
        setup_smbh : SetupSMBH
            The setup of the super-massive black hole modeling (e.g. its `MassProfile` and if its centre is fixed).
        """

        self.path_prefix = path_prefix
        self.redshift_galaxy = redshift_galaxy
        self.setup_hyper = setup_hyper
        self.setup_light = setup_light
        self.setup_mass = setup_mass
        self.setup_smbh = setup_smbh

        if isinstance(self.setup_light, SetupLightParametric) and isinstance(
            self.setup_mass, SetupMassLightDark
        ):
            self.setup_mass.bulge_prior_model = self.setup_light.bulge_prior_model
            self.setup_mass.disk_prior_model = self.setup_light.disk_prior_model
            self.setup_mass_envelope_prior_model = self.setup_light.envelope_prior_model
