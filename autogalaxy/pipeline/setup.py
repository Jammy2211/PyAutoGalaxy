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

    @property
    def tag(self):
        raise NotImplementedError


class SetupHyper(AbstractSetup):
    def __init__(
        self,
        hyper_galaxies: bool = False,
        hyper_image_sky: Optional[type(hd.HyperImageSky)] = None,
        hyper_background_noise: Optional[type(hd.HyperBackgroundNoise)] = None,
        hyper_search_no_inversion: Optional[af.NonLinearSearch] = None,
        hyper_search_with_inversion: Optional[af.NonLinearSearch] = None,
        evidence_tolerance: Optional[float] = None,
    ):
        """
        The hyper setup of a pipeline, which controls how hyper-features in PyAutoGalaxy template pipelines run,
        for example controlling whether hyper galaxies are used to scale the noise and the non-linear searches used
        in these phases.

        Users can write their own pipelines which do not use or require the *SetupHyper* class.

        This class enables pipeline tagging, whereby the hyper setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

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
            if (
                hyper_search_no_inversion is not None
                or hyper_search_with_inversion is not None
            ):
                raise exc.PipelineException(
                    "You have manually specified a search in the SetupPipeline, and an evidence_tolerance."
                    "You cannot manually specify both - remove one."
                    "(If you want the hyper search to use a specific evidence tolerance, include the evidence"
                    "tolerance in its parameters"
                )

        self.hyper_galaxies = hyper_galaxies

        self.hyper_galaxy_names = None

        if hyper_search_no_inversion is None:
            self.hyper_search_no_inversion = af.DynestyStatic(
                n_live_points=50,
                evidence_tolerance=self.evidence_tolerance,
                sample="rstagger",
            )
        elif hyper_search_no_inversion is not None:
            self.hyper_search_no_inversion = hyper_search_no_inversion

        if hyper_search_with_inversion is None:
            self.hyper_search_with_inversion = af.DynestyStatic(
                n_live_points=50,
                evidence_tolerance=self.evidence_tolerance,
                sample="rstagger",
            )
        else:
            self.hyper_search_with_inversion = hyper_search_with_inversion

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

    @property
    def component_name(self) -> str:
        """
        The name of the hyper component of a `hyper` pipeline which preceeds the `Setup` tag contained within square
        brackets.

        For the default configuration files in `config/notation/setup_tags.ini` this tag appears as `hyper[tag]`.

        Returns
        -------
        str
            The component name of the hyper pipeline.
        """
        return conf.instance["notation"]["setup_tags"]["names"]["hyper"]

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

    @property
    def tag(self):
        """
        Tag the pipeline according to the setup of the hyper features, which customizes the pipeline output paths.

        This includes tags for whether hyper-galaxies are used to scale the noise-map and whether the background sky or
        noise are fitted for by the pipeline.

        For the default configuration files in `config/notation/setup_tags.ini` example tags appear as:

        - hyper[galaxies__bg_sky]
        - hyper[bg_sky__bg_noise]
        """
        if self.hypers_all_off:
            return ""

        return (
            f"{self.component_name}["
            f"{self.hyper_galaxies_tag}"
            f"{self.hyper_image_sky_tag}"
            f"{self.hyper_background_noise_tag}]"
        )

    @property
    def hyper_galaxies_tag(self) -> str:
        """
        Tag for if hyper-galaxies are used in a hyper pipeline to scale the noise-map during model fitting, which
        customizes the pipeline's output paths.

        This tag depends on the `hyper_galaxies` bool of the `SetupHyper`.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - `hyper_galaxies=`False` -> No Tag
        - `hyper_galaxies=`True` -> hyper[galaxies]

        This is used to generate an overall tag in `tag`.
        """
        if not self.hyper_galaxies:
            return ""
        return conf.instance["notation"]["setup_tags"]["hyper"]["hyper_galaxies"]

    @property
    def hyper_image_sky_tag(self) -> str:
        """
        Tag for if the sky-background is a fitted for as a parameter in a hyper pipeline, which customizes the
        pipeline' output paths.

        This tag depends on the `hyper_image_sky` bool of the `SetupHyper`.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - `hyper_image_sky=`False` -> No Tag
        - `hyper_image_sky=`True` -> hyper[bg_sky]

        This is used to generate an overall tag in `tag`.
        """
        if self.hyper_image_sky is None:
            return ""
        return (
            f"__{conf.instance['notation']['setup_tags']['hyper']['hyper_image_sky']}"
        )

    @property
    def hyper_background_noise_tag(self) -> str:
        """
        Tag for if the background noise level is a fitted for as a parameter in a hyper pipeline, which customizes the
        pipeline' output paths.

        This tag depends on the `hyper_background_noise` bool of the `SetupHyper`.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - `hyper_background_noise=None` -> No Tag
        - `hyper_background_noise=`True` -> hyper[bg_noise]

        This is used to generate an overall tag in `tag`.
        """
        if self.hyper_background_noise is None:
            return ""
        return f"__{conf.instance['notation']['setup_tags']['hyper']['hyper_background_noise']}"

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
    @property
    def component_name(self) -> str:
        """
        The name of the light component of a `Light` pipeline which preceeds the `Setup` tag contained within square
        brackets.

        For the default configuration files this tag appears as `light[tag]`.

        Returns
        -------
        str
            The component name of the light pipeline.
        """
        return conf.instance["notation"]["setup_tags"]["names"]["light"]


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
    ):
        """
        The setup of the parametric light modeling in a pipeline, which controls how PyAutoGalaxy template pipelines
        run, for example controlling assumptions about the bulge-disk model.

        Users can write their own pipelines which do not use or require the `SetupLightParametric` class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

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
        )

    @property
    def tag(self):
        """
        Tag the pipeline according to the setup of the light pipeline which customizes the pipeline output paths.

        This includes tags for the `LightProfile` `PriorModel`'s and the alignment of different components in the model.

        For the default configuration files in `config/notation/setup_tags.ini` example tags appear as:

        - light[bulge_sersic__align_bulge_disk_centre]
        - light[bulge_core_sersic__disk_exp__envelope_exp]
        """
        return (
            f"{self.component_name}[parametric"
            f"{self.bulge_prior_model_tag}"
            f"{self.disk_prior_model_tag}"
            f"{self.envelope_prior_model_tag}"
            f"{self.align_bulge_disk_tag}"
            f"{self.align_bulge_envelope_centre_tag}"
            f"{self.light_centre_tag}]"
        )

    @property
    def bulge_prior_model_tag(self) -> str:
        """
        The tag of the bulge `PriorModel` the `LightProfile` class given to the `bulge_prior_model`.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - `EllipticalSersic` -> sersic
        - `EllipticalExponential` -> exp
        - `SphericalSersic` -> sersic_sph

        Returns
        -------
        str
            The tag of the bulge prior model.
        """

        if self.bulge_prior_model is None:
            return ""

        tag = conf.instance["notation"]["prior_model_tags"]["light"][
            self.bulge_prior_model.name
        ]

        return f"__bulge_{tag}"

    @property
    def disk_prior_model_tag(self) -> str:
        """
        The tag of the disk `PriorModel` the `LightProfile` class given to the `disk_prior_model`.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - `EllipticalSersic` -> sersic
        - `EllipticalExponential` -> exp
        - `SphericalSersic` -> sersic_sph

        Returns
        -------
        str
            The tag of the disk prior model.
        """

        if self.disk_prior_model is None:
            return ""

        tag = conf.instance["notation"]["prior_model_tags"]["light"][
            self.disk_prior_model.name
        ]

        return f"__disk_{tag}"

    @property
    def envelope_prior_model_tag(self) -> str:
        """
        The tag of the envelope `PriorModel` the `LightProfile` class given to the `envelope_prior_model`.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - `EllipticalSersic` -> sersic
        - `EllipticalExponential` -> exp
        - `SphericalSersic` -> sersic_sph

        Returns
        -------
        str
            The tag of the envelope prior model.

        """

        if self.envelope_prior_model is None:
            return ""

        tag = conf.instance["notation"]["prior_model_tags"]["light"][
            self.envelope_prior_model.name
        ]

        return f"__envelope_{tag}"

    @property
    def light_centre_tag(self) -> str:
        """
        The tag for whether the centre of the lens light `PriorModel`'s of the pipeline are fixed to an input value
        to customize pipeline output paths.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        light_centre = None -> setup
        light_centre = (1.0, 1.0) -> setup___light_centre_(1.0, 1.0)
        light_centre = (3.0, -2.0) -> setup___light_centre_(3.0, -2.0)
        """
        if self.light_centre is None:
            return ""
        else:
            y = "{0:.2f}".format(self.light_centre[0])
            x = "{0:.2f}".format(self.light_centre[1])
            return f"__{conf.instance['notation']['setup_tags']['light']['light_centre']}_({y},{x})"

    @property
    def align_bulge_disk_centre_tag(self) -> str:
        """
        Tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize pipeline
        output paths based on the bulge-disk model.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - align_bulge_disk_centre = `False` -> No Tag
        - align_bulge_disk_centre = `True `-> align_bulge_disk_centre
        """
        if self.bulge_prior_model is None or self.disk_prior_model is None:
            return ""

        if not self.align_bulge_disk_centre:
            return ""
        elif self.align_bulge_disk_centre:
            return f"_{conf.instance['notation']['setup_tags']['light']['align_bulge_disk_centre']}"

    @property
    def align_bulge_disk_elliptical_comps_tag(self) -> str:
        """
        Tag if the elliptical components of the bulge and disk `PriorModel`s are aligned or not, to customize pipeline
        output paths based on the bulge-disk model.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - align_bulge_disk_elliptical_comps = `False` -> No Tag
        - align_bulge_disk_elliptical_comps = `True `-> align_bulge_disk_ell
        """
        if self.bulge_prior_model is None or self.disk_prior_model is None:
            return ""

        if not self.align_bulge_disk_elliptical_comps:
            return ""
        elif self.align_bulge_disk_elliptical_comps:
            return f"_{conf.instance['notation']['setup_tags']['light']['align_bulge_disk_elliptical_comps']}"

    @property
    def align_bulge_envelope_centre_tag(self) -> str:
        """
        Tag for if the bulge and envelope of a bulge-envelope system are aligned or not, to customize pipeline
        output paths based on the bulge-envelope model.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - align_bulge_envelope_centre = `False` -> No Tag
        - align_bulge_envelope_centre = `True `-> align_bulge_envelope_centre
        """
        if self.bulge_prior_model is None or self.envelope_prior_model is None:
            return ""

        if not self.align_bulge_envelope_centre:
            return ""
        elif self.align_bulge_envelope_centre:
            return f"_{conf.instance['notation']['setup_tags']['light']['align_bulge_envelope_centre']}"

    @property
    def align_bulge_disk_tag(self) -> str:
        """
        Tag the alignment of the the bulge and disk `PriorModel`'s, to customize  pipeline output
        paths based on the bulge-disk model.

        This adds the bulge_disk tags generated in the functions `align_bulge_disk_centre_tag` and
        `align_bulge_disk_elliptical_comps_tag`.

        For the the default configuration files `config/notation/setup_tags.ini` example tags are:

        - align_bulge_disk_ell
        - align_bulge_disk_centre_ell
        """

        align_bulge_disk_tag = f"{self.align_bulge_disk_centre_tag}{self.align_bulge_disk_elliptical_comps_tag}"

        if align_bulge_disk_tag == "":
            return ""

        return f"__{conf.instance['notation']['setup_tags']['light']['align_bulge_disk']}{align_bulge_disk_tag}"


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

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

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
    def tag(self) -> str:
        """
        Tag the pipeline according to the setup of the `Inversion` used in a light pipeline which customizes the
        pipeline output paths.

        This includes tags for the `Pixelization` `PriorModel` and `Regularization` `PriorModel`.

        For the default configuration files in `config/notation/setup_tags.ini` example tags appear as:

        - light[pix_rect__reg_const]
        - light[pix_voro_image_1200__reg_adapt_bright]
        """
        return f"{self.component_name}[inversion{self.pixelization_tag}{self.inversion_pixels_fixed_tag}{self.regularization_tag}]"

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

    @property
    def inversion_pixels_fixed_tag(self) -> str:
        """
        The tag for the number of fixed pixels used by the `Inversion` if an fixed input value is used.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - `inversion_pixels_fixed=None` -> No Tag
        - `inversion_pixels_fixed=1200` -> '_1200'

        Returns
        -------
        str
            The tag of the number of fixed inversion pixels.
        """
        if (
            self.inversion_pixels_fixed is None
            or self._pixelization_prior_model.cls is not pix.VoronoiBrightnessImage
        ):
            return ""

        return f"_{str(self.inversion_pixels_fixed)}"

    @property
    def pixelization_tag(self) -> str:
        """
        The tag for the `Pixelization` `PriorModel` used by the light `Inversion` pipeline.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        `pixelization = af.PriorModel(pix.Rectangular)` -> light[pix_rect]
        `pixelization = af.PriorModel(pix.VoronoiMagnification)` -> light[pix_voro_mag]
        """
        return (
            f"__{conf.instance['notation']['setup_tags']['inversion']['pixelization']}_"
            f"{conf.instance['notation']['prior_model_tags']['pixelization'][self._pixelization_prior_model.name]}"
        )

    @property
    def regularization_tag(self) -> str:
        """
        The tag for the `Regularization` `PriorModel` used by the light `Inversion` pipeline.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        `regularization = reg.Constant` -> light[reg_const]
        `regularization = reg.AdaptiveBrightness` -> light[reg_adapt_bright]
        """
        return (
            f"__{conf.instance['notation']['setup_tags']['inversion']['regularization']}_"
            f"{conf.instance['notation']['prior_model_tags']['regularization'][self.regularization_prior_model.name]}"
        )


class AbstractSetupMass(AbstractSetup):
    def __init__(self, mass_centre: (float, float) = None):
        """
        The setup of the mass modeling in a pipeline for `MassProfile`'s representing the total (e.g. stars + dark
        matter) mass distribution, which controls how PyAutoGalaxy template pipelines run, for example controlling
        assumptions about the bulge-disk model.

        Users can write their own pipelines which do not use or require the `` class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        mass_centre : (float, float)
           If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
           non-linear search.
        """

        self.mass_centre = mass_centre

    @property
    def component_name(self) -> str:
        """
        The name of the mass component of a `mass` pipeline which preceeds the `Setup` tag contained within square
        brackets.

        For the default configuration files this tag appears as `mass[tag]`.

        Returns
        -------
        str
            The component name of the mass pipeline.
        """
        return conf.instance["notation"]["setup_tags"]["names"]["mass"]

    @property
    def mass_centre_tag(self) -> str:
        """
        The tag for whether the centre of the lens mass `PriorModel`'s of the pipeline are fixed to an input value
        to customize pipeline output paths.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        mass_centre = None -> setup
        mass_centre = (1.0, 1.0) -> setup___mass_centre_(1.0, 1.0)
        mass_centre = (3.0, -2.0) -> setup___mass_centre_(3.0, -2.0)
        """
        if self.mass_centre is None:
            return ""

        y = "{0:.2f}".format(self.mass_centre[0])
        x = "{0:.2f}".format(self.mass_centre[1])
        return f"__{conf.instance['notation']['setup_tags']['mass']['mass_centre']}_({y},{x})"

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

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

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

    @property
    def tag(self):
        """
        Tag the pipeline according to the setup of the total mass pipeline which customizes the pipeline output paths.

        This includes tags for the `MassProfile` `PriorModel`'s and the alignment of different components in the model.

        For the default configuration files in `config/notation/setup_tags.ini` example tags appear as:

        - mass[total__sie]
        - mass[total__power_law__centre_(0.0,0.0)]
        """
        return (
            f"{self.component_name}[total"
            f"{self.mass_prior_model_tag}"
            f"{self.mass_centre_tag}"
            f"{self.align_bulge_mass_centre_tag}]"
        )

    @property
    def mass_prior_model_tag(self) -> str:
        """
        The tag of the mass `PriorModel` the `MassProfile` class given to the `mass_prior_model`.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - `EllipticalIsothermal` -> sie
        - `EllipticalPowerLaw` -> power_law

        Returns
        -------
        str
            The tag of the mass prior model.
        """

        if self.mass_prior_model is None:
            return ""

        return f"__{conf.instance['notation']['prior_model_tags']['mass'][self.mass_prior_model.name]}"

    @property
    def align_bulge_mass_centre_tag(self) -> str:
        """
        Tags if the lens mass model centre is aligned with that of its light profile.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        align_bulge_mass_centre = `False` -> setup
        align_bulge_mass_centre = `True` -> setup___align_bulge_mass_centre
        """
        if self.mass_centre is not None:
            return ""

        if not self.align_bulge_mass_centre:
            return ""
        return f"__{conf.instance['notation']['setup_tags']['mass']['align_bulge_mass_centre']}"

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

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

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

    @property
    def tag(self):
        """
        Tag the pipeline according to the setup of the decomposed light and dark mass pipeline which customizes
        the pipeline output paths.

        This includes tags for the `MassProfile` `PriorModel`'s and the alignment of different components in the model.

        For the default configuration files in `config/notation/setup_tags.ini` example tags appear as:

        - mass[light_dark__bulge_]
        - mass[total_power_law__centre_(0.0,0.0)]
        """
        return (
            f"{self.component_name}[light_dark"
            f"{self.bulge_prior_model_tag}"
            f"{self.disk_prior_model_tag}"
            f"{self.envelope_prior_model_tag}"
            f"{self.constant_mass_to_light_ratio_tag}"
            f"{self.dark_prior_model_tag}"
            f"{self.mass_centre_tag}"
            f"{self.align_bulge_dark_centre_tag}]"
        )

    @property
    def constant_mass_to_light_ratio_tag(self) -> str:
        """Generate a tag for whether the mass-to-light ratio in a light-dark mass model is constaant (shared amongst
         all light and mass profiles) or free (all mass-to-light ratios are free parameters).

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        constant_mass_to_light_ratio = `False` -> mlr_free
        constant_mass_to_light_ratio = `True` -> mlr_constant
        """
        if self.constant_mass_to_light_ratio:
            return f"__mlr_{conf.instance['notation']['setup_tags']['mass']['constant_mass_to_light_ratio']}"
        return f"__mlr_{conf.instance['notation']['setup_tags']['mass']['free_mass_to_light_ratio']}"

    @property
    def bulge_prior_model_tag(self) -> str:
        """
        The tag of the bulge `PriorModel` using the tags specified in the setup_tags.ini config file.

        This tag depends on the `LightMassProfile` class given to the bulge, for example for the default configuration
        files:

        - `EllipticalSersic` -> sersic
        - `EllipticalExponential` -> exp
        - `SphericalSersic` -> sersic_sph

        Returns
        -------
        str
            The tag of the bulge prior model.

        """

        if self.bulge_prior_model is None:
            return ""

        tag = conf.instance["notation"]["prior_model_tags"]["mass"][
            self.bulge_prior_model.name
        ]

        return f"__bulge_{tag}"

    @property
    def disk_prior_model_tag(self) -> str:
        """
        The tag of the disk `PriorModel` using the tags specified in the setup_tags.ini config file.

        This tag depends on the `LightMassProfile` class given to the disk, for example for the default configuration
        files:

        - `EllipticalSersic` -> sersic
        - `EllipticalExponential` -> exp
        - `SphericalSersic` -> sersic_sph

        Returns
        -------
        str
            The tag of the disk prior model.
        """

        if self.disk_prior_model is None:
            return ""

        tag = conf.instance["notation"]["prior_model_tags"]["mass"][
            self.disk_prior_model.name
        ]

        return f"__disk_{tag}"

    @property
    def envelope_prior_model_tag(self) -> str:
        """
        The tag of the envelope `PriorModel` using the tags specified in the setup_tags.ini config file.

        This tag depends on the `LightMassProfile` class given to the envelope, for example for the default
        configuration files:

        - `EllipticalSersic` -> sersic
        - `EllipticalExponential` -> exp
        - `SphericalSersic` -> sersic_sph

        Returns
        -------
        str
            The tag of the envelope prior model.

        """

        if self.envelope_prior_model is None:
            return ""

        tag = conf.instance["notation"]["prior_model_tags"]["mass"][
            self.envelope_prior_model.name
        ]

        return f"__envelope_{tag}"

    @property
    def dark_prior_model_tag(self) -> str:
        """
        The tag of the dark `PriorModel` using the tags specified in the setup_tags.ini config file.

        This tag depends on the `MassProfile` class given to the dark, for example for the default
        configuration files:

        - `SphericalNFWMCRLudlow` -> nfw_sph_ludlow
        - `EllipticalNFW` -> nfw
        - `SphericalTruncatedNFW` -> nfw_trun_sph

        Returns
        -------
        str
            The tag of the dark prior model.
        """

        if self.dark_prior_model is None:
            return ""

        tag = conf.instance["notation"]["prior_model_tags"]["mass"][
            self.dark_prior_model.name
        ]

        return f"__dark_{tag}"

    @property
    def align_bulge_dark_centre_tag(self) -> str:
        """
        Tag for if the lens mass model's bulge `PriorModel` is aligned with the dark matter centre.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        align_bulge_dark_centre = `False` -> mass[light_dark]
        align_bulge_dark_centre = `True` -> mass[light_dark__align_bulge_dark_centre]
        """
        if not self.align_bulge_dark_centre:
            return ""
        return f"__{conf.instance['notation']['setup_tags']['mass']['align_bulge_dark_centre']}"

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

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

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

    @property
    def component_name(self) -> str:
        """
        The name of the smbh component of a `smbh` pipeline which preceeds the `Setup` tag contained within square
        brackets.

        For the default configuration files this tag appears as `smbh[tag]`.

        Returns
        -------
        str
            The component name of the smbh pipeline.
        """
        return conf.instance["notation"]["setup_tags"]["names"]["smbh"]

    @property
    def tag(self):
        return (
            f"{self.component_name}[{self.smbh_prior_model_tag}{self.smbh_centre_tag}]"
        )

    @property
    def smbh_prior_model_tag(self) -> str:
        """
        The tag of the smbh `PriorModel` the `MassProfile` class given to the `smbh_prior_model`.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        - `EllipticalIsothermal` -> sie
        - `EllipticalPowerLaw` -> power_law

        Returns
        -------
        str
            The tag of the smbh prior model.
        """
        return f"{conf.instance['notation']['prior_model_tags']['mass'][self.smbh_prior_model.name]}"

    @property
    def smbh_centre_tag(self) -> str:
        """
        Tag whether the smbh `PriorModel` centre is fixed or fitted for as a free parameter.

        For the the default configuration files `config/notation/setup_tags.ini` tagging is performed as follows:

        smbh_centre_fixed=True -> smbh[centre_fixed]
        smbh_centre_fixed=False -> smbh[centre_free]
        """

        if self.smbh_centre_fixed:

            smbh_centre_tag = conf.instance["notation"]["setup_tags"]["smbh"][
                "smbh_centre_fixed"
            ]

        else:

            smbh_centre_tag = conf.instance["notation"]["setup_tags"]["smbh"][
                "smbh_centre_free"
            ]

        return f"__{smbh_centre_tag}"

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

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        path_prefix : str or None
            The prefix of folders between the output path of the pipeline and the pipeline name, tags and phase folders.
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

    def _pipeline_tag_from_setup(self, setup: AbstractSetup) -> str:
        """
        Returns the tag of a pipeline from the tag of an indiviual `Setup` object.

        Parameters
        ----------
        setup_tag : str
            The tag of the `Setup` object which is converted to the pipeline tag.

        Returns
        -------
        str
            The pipeline tag.
        """
        return f"__{setup.tag}" if setup is not None else ""

    @property
    def tag(self) -> str:
        """
        The overall pipeline tag, which customizes the 'setup' folder the results are output to.

        For the the default configuration files `config/notation/setup_tags.ini` examples of tagging are as follows:

        - setup__hyper[galaxies__bg_noise]__light[bulge_sersic__disk__exp_light_centre_(1.00,2.00)]
        - "setup__smbh[point_mass__centre_fixed]"
        """

        setup_tag = conf.instance["notation"]["setup_tags"]["pipeline"]["pipeline"]

        hyper_tag = self._pipeline_tag_from_setup(setup=self.setup_hyper)
        light_tag = self._pipeline_tag_from_setup(setup=self.setup_light)
        mass_tag = self._pipeline_tag_from_setup(setup=self.setup_mass)
        smbh_tag = self._pipeline_tag_from_setup(setup=self.setup_smbh)

        return f"{setup_tag}{hyper_tag}{light_tag}{mass_tag}{smbh_tag}"
