from autoconf import conf
import autofit as af
from autoarray.inversion import pixelizations as pix, regularization as reg
from autogalaxy.profiles import mass_profiles as mp, light_and_mass_profiles as lmp
from autogalaxy import exc


class SetupHyper:
    def __init__(
        self,
        hyper_galaxies: bool = False,
        hyper_image_sky: bool = False,
        hyper_background_noise: bool = False,
        hyper_galaxy_phase_first: bool = False,
        hyper_fixed_after_source=False,
        hyper_galaxies_search: af.NonLinearSearch = None,
        inversion_search: af.NonLinearSearch = None,
        hyper_combined_search: af.NonLinearSearch = None,
        evidence_tolerance: float = None,
    ):
        """The hyper setup of a pipeline, which controls how hyper-features in PyAutoGalaxy template pipelines run,
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
        hyper_galaxy_phase_first : bool
            If True, the hyper-galaxy phase which scales the noise map is performed before the inversion phase, else
            it is performed after.
        hyper_galaxies_search : af.NonLinearSearch or None
            The non-linear search used by every hyper-galaxies phase.
        inversion_search : af.NonLinearSearch or None
            The non-linear search used by every inversion phase.
        hyper_combined_search : af.NonLinearSearch or None
            The non-linear search used by every hyper combined phase.
        evidence_tolerance : float
            The evidence tolerance of the non-linear searches used in the hyper phases, whereby higher values will
            lead them to end earlier at the expense of accuracy.
        """

        self.evidence_tolerance = evidence_tolerance

        if evidence_tolerance is not None:
            if (
                hyper_galaxies_search is not None
                or inversion_search is not None
                or hyper_combined_search is not None
            ):
                raise exc.PipelineException(
                    "You have manually specified a search in the SetupPipeline, and an evidence_tolerance."
                    "You cannot manually specify both - remove one."
                    "(If you want the hyper search to use a specific evidence tolerance, include the evidence"
                    "tolerance in its parameters"
                )

        self.hyper_galaxies = hyper_galaxies

        if self.hyper_galaxies and hyper_galaxies_search is None:
            self.hyper_galaxies_search = af.DynestyStatic(n_live_points=75)
        elif self.hyper_galaxies and hyper_galaxies_search is not None:
            self.hyper_galaxies_search = hyper_galaxies_search
        else:
            self.hyper_galaxies_search = None

        if inversion_search is None:
            self.inversion_search = af.DynestyStatic(
                n_live_points=50,
                evidence_tolerance=self.evidence_tolerance,
                sample="rstagger",
            )
        elif inversion_search is not None:
            self.inversion_search = inversion_search

        if hyper_combined_search is None:
            self.hyper_combined_search = af.DynestyStatic(
                n_live_points=50,
                evidence_tolerance=self.evidence_tolerance,
                sample="rstagger",
            )
        else:
            self.hyper_combined_search = hyper_combined_search

        self.hyper_fixed_after_source = hyper_fixed_after_source
        self.hyper_galaxy_phase_first = hyper_galaxy_phase_first

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

    @property
    def tag(self):
        """Tag ithe hyper pipeline features used in a hyper pipeline to customize pipeline output paths.
        """
        if not any(
            [self.hyper_galaxies, self.hyper_image_sky, self.hyper_background_noise]
        ):
            return ""

        return (
            f"{conf.instance.setup_tag.get('hyper', 'hyper')}["
            f"{self.hyper_galaxies_tag}"
            f"{self.hyper_image_sky_tag}"
            f"{self.hyper_background_noise_tag}"
            f"{self.hyper_fixed_after_source_tag}]"
        )

    @property
    def hyper_galaxies_tag(self):
        """Tag if hyper-galaxies are used in a hyper pipeline to customize pipeline output paths.

        This is used to generate an overall hyper tag in *hyper_tag*.
        """
        if not self.hyper_galaxies:
            return ""
        elif self.hyper_galaxies:
            return conf.instance.setup_tag.get("hyper", "hyper_galaxies")

    @property
    def hyper_image_sky_tag(self):
        """Tag if the sky-background is hyper as a hyper_galaxies-parameter in a hyper pipeline to customize pipeline
        output paths.

        This is used to generate an overall hyper tag in *hyper_tag*.
        """
        if not self.hyper_image_sky:
            return ""
        elif self.hyper_image_sky:
            return f"_{conf.instance.setup_tag.get('hyper', 'hyper_image_sky')}"

    @property
    def hyper_background_noise_tag(self):
        """Tag if the background noise is hyper as a hyper_galaxies-parameter in a hyper pipeline to customize pipeline
        output paths.

        This is used to generate an overall hyper tag in *hyper_tag*.
        """
        if not self.hyper_background_noise:
            return ""
        elif self.hyper_background_noise:
            return f"_{conf.instance.setup_tag.get('hyper', 'hyper_background_noise')}"

    @property
    def hyper_fixed_after_source_tag(self):
        """Generate a tag for if the hyper parameters are held fixed after the source pipeline.

        This changes the pipeline setup tag as follows:

        hyper_fixed_after_source = False -> setup
        hyper_fixed_after_source = True -> setup__hyper_fixed
        """
        if not self.hyper_fixed_after_source:
            return ""
        elif self.hyper_fixed_after_source:
            return "_" + conf.instance.setup_tag.get(
                "hyper", "hyper_fixed_after_source"
            )


class AbstractSetupLight:
    def __init__(self, light_centre: (float, float) = None):
        """The setup of the light modeling in a pipeline, which controls how PyAutoGalaxy template pipelines runs.

        Users can write their own pipelines which do not use or require the *SetupLight* class.

        Parameters
        ----------
        light_centre : (float, float) or None
           If input, a fixed (y,x) centre of the galaxy is used for the light profile model which is not treated as a
            free parameter by the non-linear search.
        """
        self.light_centre = light_centre

    @property
    def light_centre_tag(self):
        """Tag if the lens light of the pipeline are fixed to a previous estimate, or varied \
         during the analysis, to customize pipeline output paths.

        This changes the setup folder as follows:

        light_centre = None -> setup
        light_centre = (1.0, 1.0) -> setup___light_centre_(1.0, 1.0)
        light_centre = (3.0, -2.0) -> setup___light_centre_(3.0, -2.0)
        """
        if self.light_centre is None:
            return ""
        else:
            y = "{0:.2f}".format(self.light_centre[0])
            x = "{0:.2f}".format(self.light_centre[1])
            return f"__{conf.instance.setup_tag.get('light', 'light_centre')}_({y},{x})"

    def align_centre_to_light_centre(self, light):
        """
        Align the centre of an input light profile to the light_centre of this instance of the SLaM Source
        class, make the centre of the light profile fixed and thus not free parameters that are fitted for.

        If the light_centre is not input (and thus None) the light profile centre is unchanged.

        Parameters
        ----------
        light : ag.mp.MassProfile
            The light profile whose centre may be aligned with the light_centre attribute.
        """
        if self.light_centre is not None:
            light.centre = self.light_centre
        return light

    def unalign_mass_centre_from_light_centre(self, mass):
        """If the centre of a mass model was previously aligned with that of the lens light centre, unaligned them
        by using an earlier model of the light.
        """
        if self.align_light_mass_centre:

            mass.centre = af.last[-3].model.galaxies.lens.bulge.centre

        else:

            mass.centre = af.last[-1].model.galaxies.lens.mass.centre

        return mass


class SetupLightSersic(AbstractSetupLight):
    def __init__(self, light_centre: (float, float) = None):
        """The setup of the light modeling in a pipeline, which controls how PyAutoGalaxy template pipelines runs, for
        example controlling assumptions about the bulge-disk model.

        Users can write their own pipelines which do not use or require the *SetupLight* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        light_centre : (float, float) or None
           If input, a fixed (y,x) centre of the galaxy is used for the light profile model which is not treated as a
            free parameter by the non-linear search.
        """

        super().__init__(light_centre=light_centre)

    @property
    def model_type(self):
        return "bulge"

    @property
    def tag(self):
        """Generate a tag of the parameetric source model.
        """
        return (
            f"{conf.instance.setup_tag.get('light', 'light')}[{self.model_type}"
            f"{self.light_centre_tag}]"
        )


class SetupLightBulgeDisk(AbstractSetupLight):
    def __init__(
        self,
        light_centre: (float, float) = None,
        align_bulge_disk_centre: bool = False,
        align_bulge_disk_elliptical_comps: bool = False,
        disk_as_sersic: bool = False,
    ):
        """The setup of the light modeling in a pipeline, which controls how PyAutoGalaxy template pipelines runs, for
        example controlling assumptions about the bulge-disk model.

        Users can write their own pipelines which do not use or require the *SetupLight* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        light_centre : (float, float) or None
           If input, a fixed (y,x) centre of the galaxy is used for the light profile model which is not treated as a
            free parameter by the non-linear search.
        align_bulge_disk_centre : bool or None
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will align the centre of the bulge and disk components and not fit them separately.
        align_bulge_disk_elliptical_comps : bool or None
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will align the elliptical components the bulge and disk components and not fit them separately.
        disk_as_sersic : bool or None
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will use an EllipticalSersic for the disk instead of an EllipticalExponential.
        """

        super().__init__(light_centre=light_centre)

        self.align_bulge_disk_centre = align_bulge_disk_centre
        self.align_bulge_disk_elliptical_comps = align_bulge_disk_elliptical_comps
        self.disk_as_sersic = disk_as_sersic

    @property
    def model_type(self):
        return "bulge_disk"

    @property
    def tag(self):
        return (
            f"{conf.instance.setup_tag.get('light', 'light')}[{self.model_type}"
            f"{self.light_centre_tag}"
            f"{self.align_bulge_disk_tag}"
            f"{self.disk_as_sersic_tag}]"
        )

    @property
    def align_bulge_disk_centre_tag(self):
        """Tag if the bulge and disk of a bulge-disk system are aligned or not, to customize pipeline
        output paths based on the bulge-disk model.

        This is used to generate an overall align-bulge disk tag in *align_bulge_disk_tag*.
        """
        if not self.align_bulge_disk_centre:
            return ""
        elif self.align_bulge_disk_centre:
            return f"_{conf.instance.setup_tag.get('light', 'align_bulge_disk_centre')}"

    @property
    def align_bulge_disk_elliptical_comps_tag(self):
        """Tag if the ellipticity of the bulge and disk of a bulge-disk system are aligned or not, to customize pipeline
        output paths based on the bulge-disk model.

        This is used to generate an overall align-bulge disk tag in *align_bulge_disk_tag*.
        """
        if not self.align_bulge_disk_elliptical_comps:
            return ""
        elif self.align_bulge_disk_elliptical_comps:
            return f"_{conf.instance.setup_tag.get('light', 'align_bulge_disk_elliptical_comps')}"

    @property
    def align_bulge_disk_tag(self):
        """Tag the alignment of the geometry of the bulge and disk of a bulge-disk system, to customize  pipeline output
        paths based on the bulge-disk model. This adds together the bulge_disk tags generated in the  3 functions
        *align_bulge_disk_centre_tag*, *align_bulge_disk_axis_ratio_tag* and *align_bulge_disk_phi_tag*.
        """

        if not any(
            [self.align_bulge_disk_centre, self.align_bulge_disk_elliptical_comps]
        ):
            return ""

        return f"__{conf.instance.setup_tag.get('light', 'align_bulge_disk')}{self.align_bulge_disk_centre_tag}{self.align_bulge_disk_elliptical_comps_tag}"

    @property
    def disk_as_sersic_tag(self):
        """Tag if the disk component of a bulge-disk light profile fit of the pipeline is modeled as a EllipticalSersic
        or an EllipticalExponential.

        This changes the setup folder as follows:

        disk_as_sersic = False -> setup
        disk_as_sersic = True -> setup___disk_as_sersic
        """
        if not self.disk_as_sersic:
            return ""
        elif self.disk_as_sersic:
            return f"__{conf.instance.setup_tag.get('light', 'disk_as_sersic')}"


class AbstractSetupMass:
    def __init__(self, mass_centre: (float, float) = None):
        """The setup of mass modeling in a pipeline, which controls how PyAutoLens template pipelines runs, for
        example controlling assumptions about the mass-to-light profile used too control how a light profile is
        converted to a mass profile.

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

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
    def mass_centre_tag(self):
        """Generate a tag if the lens mass model centre of the pipeline is fixed to an input value, to customize
        pipeline output paths.

        This changes the setup folder as follows:

        mass_centre = None -> setup
        mass_centre = (1.0, 1.0) -> setup___mass_centre_(1.0, 1.0)
        mass_centre = (3.0, -2.0) -> setup___mass_centre_(3.0, -2.0)
        """
        if self.mass_centre is None:
            return ""

        y = "{0:.2f}".format(self.mass_centre[0])
        x = "{0:.2f}".format(self.mass_centre[1])
        return f"__{conf.instance.setup_tag.get('mass', 'mass_centre')}_({y},{x})"

    def align_centre_to_mass_centre(self, mass):
        """
        Align the centre of an input mass profile to the mass_centre of this instance of the SLaM Source
        class, make the centre of the mass profile fixed and thus not free parameters that are fitted for.

        If the mass_centre is not input (and thus None) the mass profile centre is unchanged.

        Parameters
        ----------
        mass : ag.mp.MassProfile
            The mass profile whose centre may be aligned with the mass_centre attribute.
        """
        if self.mass_centre is not None:
            mass.centre = self.mass_centre
        return mass

    def unfix_mass_centre(self, mass, index=0):
        """If the centre of a mass model was previously fixed to an input value (e.g. mass_centre), unaligned it
        by making its centre GaussianPriors.
        """

        if self.mass_centre is not None:

            mass.centre.centre_0 = af.GaussianPrior(
                mean=self.mass_centre[0], sigma=0.05
            )
            mass.centre.centre_1 = af.GaussianPrior(
                mean=self.mass_centre[1], sigma=0.05
            )

        else:

            mass.centre = af.last[index].model.galaxies.lens.mass.centre

        return mass


class SetupMassTotal(AbstractSetupMass):
    def __init__(
        self,
        mass_profile: mp.MassProfile = None,
        mass_centre: (float, float) = None,
        align_light_mass_centre: bool = False,
    ):
        """The setup of mass modeling in a pipeline, which controls how PyAutoLens template pipelines runs, for
        example controlling assumptions about the mass-to-light profile used too control how a light profile is
        converted to a mass profile.

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        mass_profile : mp.MassProfile
            The _MassProfile_ fitted by the _Pipeline_ (the pipeline must specifically use this option to use this
            mass profile)
        mass_centre : (float, float)
           If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
           non-linear search.
        """

        super().__init__(mass_centre=mass_centre)

        self.mass_profile = mass_profile

        self.align_light_mass_centre = align_light_mass_centre

    @property
    def model_type(self):
        return "total"

    @property
    def tag(self):
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to.
        """
        return (
            f"{conf.instance.setup_tag.get('mass', 'mass')}[{self.model_type}{self.mass_profile_tag}"
            f"{self.mass_centre_tag}"
            f"{self.align_light_mass_centre_tag}]"
        )

    @property
    def mass_profile_tag(self):

        if self.mass_profile is None:
            return ""

        return f"__{conf.instance.setup_tag.get('total', self.mass_profile().__class__.__name__)}"

    @property
    def align_light_mass_centre_tag(self):
        """Generate a tag if the lens mass model is centre is aligned with that of its light profile.

        This changes the setup folder as follows:

        align_light_mass_centre = False -> setup
        align_light_mass_centre = True -> setup___align_light_mass_centre
        """
        if self.mass_centre is not None:
            return ""

        if not self.align_light_mass_centre:
            return ""
        return f"__{conf.instance.setup_tag.get('mass', 'align_light_mass_centre')}"

    def align_centre_of_mass_to_light(self, mass, light_centre):
        """Align the centre of a mass profile to the centre of a light profile, if the align_light_mass_centre
        SLaM setting is True.

        Parameters
        ----------
        mass : ag.mp.MassProfile
            The mass profile whose centre may be aligned with the light_centre attribute.
        light : (float, float)
            The centre of the light profile the mass profile is aligned with.
        """
        if self.align_light_mass_centre:
            mass.centre = light_centre
        else:
            mass.centre.centre_0 = af.GaussianPrior(mean=light_centre[0], sigma=0.1)
            mass.centre.centre_1 = af.GaussianPrior(mean=light_centre[1], sigma=0.1)
        return mass


class SetupMassLightDark(AbstractSetupMass):
    def __init__(
        self,
        mass_centre: (float, float) = None,
        constant_mass_to_light_ratio: bool = False,
        bulge_mass_to_light_ratio_gradient: bool = False,
        disk_mass_to_light_ratio_gradient: bool = False,
        align_light_dark_centre: bool = False,
        align_bulge_dark_centre: bool = False,
    ):
        """The setup of mass modeling in a pipeline, which controls how PyAutoLens template pipelines runs, for
        example controlling assumptions about the mass-to-light profile used too control how a light profile is
        converted to a mass profile.

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        mass_centre : (float, float)
           If input, a fixed (y,x) centre of the mass profile is used which is not treated as a free parameter by the
           non-linear search.
        constant_mass_to_light_ratio : bool
            If True, and the mass model consists of multiple _LightProfile_ and _MassProfile_ coomponents, the
            mass-to-light ratio's of all components are fixed to one shared value.
        bulge_mass_to_light_ratio_gradient : bool
            If True, the bulge _EllipticalSersic_ component of the mass model is altered to include a gradient in its
            mass-to-light ratio conversion.
        disk_mass_to_light_ratio_gradient : bool
            If True, the bulge _EllipticalExponential_ component of the mass model is altered to include a gradient in
            its mass-to-light ratio conversion.
        align_light_dark_centre : bool
            If True, and the mass model is a sersic and dark matter model (e.g. EllipticalSersic + SphericalNFW),
            the centre of the bulge and dark matter profiles are aligned.
        align_bulge_mass_centre : bool
            If True, and the mass model is a decomposed bulge, disk and dark matter model (e.g. EllipticalSersic +
            EllipticalExponential + SphericalNFW), the centre of the bulge and dark matter profiles are aligned.
        """

        super().__init__(mass_centre=mass_centre)

        if align_light_dark_centre and align_bulge_dark_centre:
            raise exc.SetupException(
                "In PipelineMassSettings align_light_dark_centre and align_bulge_disk_centre"
                "can not both be True (one is not relevent to the light profile you are fitting"
            )

        self.constant_mass_to_light_ratio = constant_mass_to_light_ratio
        self.bulge_mass_to_light_ratio_gradient = bulge_mass_to_light_ratio_gradient
        self.disk_mass_to_light_ratio_gradient = disk_mass_to_light_ratio_gradient
        self.align_light_dark_centre = align_light_dark_centre
        self.align_bulge_dark_centre = align_bulge_dark_centre
        self.disk_as_sersic = None

    @property
    def model_type(self):
        return "light_dark"

    @property
    def tag(self):
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to.
        """
        return (
            f"{conf.instance.setup_tag.get('mass', 'mass')}[{self.model_type}"
            f"{self.mass_centre_tag}"
            f"{self.mass_to_light_tag}"
            f"{self.align_light_dark_centre_tag}"
            f"{self.align_bulge_dark_centre_tag}]"
        )

    @property
    def mass_to_light_tag(self):
        """Generate a tag about the mass-to-light conversion in the mass model, in particular:

         - Whether the mass-to-light ratio is constant (shared amongst all light and mass profiles) or free (all
           mass-to-light ratios are free parameters).
         - Whether certain components in the mass model include a gradient in their light-to-mass conversion.
        """

        mass_to_light_tag = (
            f"__{conf.instance.setup_tag.get('mass', 'mass_to_light_ratio')}"
            f"{self.constant_mass_to_light_ratio_tag}"
        )

        if (
            self.bulge_mass_to_light_ratio_gradient
            or self.disk_mass_to_light_ratio_gradient
        ):
            gradient_tag = conf.instance.setup_tag.get(
                "mass", "mass_to_light_ratio_gradient"
            )
            if self.bulge_mass_to_light_ratio_gradient:
                gradient_tag = (
                    f"{gradient_tag}{self.bulge_mass_to_light_ratio_gradient_tag}"
                )
            if self.disk_mass_to_light_ratio_gradient:
                gradient_tag = (
                    f"{gradient_tag}{self.disk_mass_to_light_ratio_gradient_tag}"
                )
            return f"{mass_to_light_tag}_{gradient_tag}"
        else:
            return mass_to_light_tag

    @property
    def constant_mass_to_light_ratio_tag(self):
        """Generate a tag for whether the mass-to-light ratio in a light-dark mass model is constaant (shared amongst
         all light and mass profiles) or free (all mass-to-light ratios are free parameters).

        This changes the setup folder as follows:

        constant_mass_to_light_ratio = False -> mlr_free
        constant_mass_to_light_ratio = True -> mlr_constant
        """
        if self.constant_mass_to_light_ratio:
            return f"_{conf.instance.setup_tag.get('mass', 'constant_mass_to_light_ratio')}"
        return f"_{conf.instance.setup_tag.get('mass', 'free_mass_to_light_ratio')}"

    @property
    def bulge_mass_to_light_ratio_gradient_tag(self):
        """Generate a tag for whether the mass-to-light ratio in a light-dark mass model is constaant (shared amongst
         all light and mass profiles) or free (all mass-to-light ratios are free parameters).

        This changes the setup folder as follows:

        constant_mass_to_light_ratio = False -> mlr_free
        constant_mass_to_light_ratio = True -> mlr_constant
        """
        if not self.bulge_mass_to_light_ratio_gradient:
            return ""
        return f"_{conf.instance.setup_tag.get('mass', 'bulge_mass_to_light_ratio_gradient')}"

    @property
    def disk_mass_to_light_ratio_gradient_tag(self):
        """Generate a tag for whether the mass-to-light ratio in a light-dark mass model is constaant (shared amongst
         all light and mass profiles) or free (all mass-to-light ratios are free parameters).

        This changes the setup folder as follows:

        constant_mass_to_light_ratio = False -> mlr_free
        constant_mass_to_light_ratio = True -> mlr_constant
        """
        if not self.disk_mass_to_light_ratio_gradient:
            return ""
        return f"_{conf.instance.setup_tag.get('mass', 'disk_mass_to_light_ratio_gradient')}"

    @property
    def align_light_dark_centre_tag(self):
        """Generate a tag if the lens mass model is a decomposed light + dark matter model if their centres are aligned.

        This changes the setup folder as follows:

        align_light_dark_centre = False -> setup
        align_light_dark_centre = True -> setup___align_light_dark_centre
        """
        if not self.align_light_dark_centre:
            return ""
        return f"__{conf.instance.setup_tag.get('mass', 'align_light_dark_centre')}"

    @property
    def align_bulge_dark_centre_tag(self):
        """Generate a tag if the lens mass model is a decomposed bulge + disk + dark matter model if the bulge centre
        is aligned with the dark matter centre.

        This changes the setup folder as follows:

        align_bulge_dark_centre = False -> setup
        align_bulge_dark_centre = True -> setup___align_bulge_dark_centre
        """
        if not self.align_bulge_dark_centre:
            return ""
        return "__" + conf.instance.setup_tag.get("mass", "align_bulge_dark_centre")

    @property
    def disk_as_sersic_tag(self):
        """Tag if the disk component of a bulge-disk light profile fit of the pipeline is modeled as a EllipticalSersic
        or an EllipticalExponential.

        This changes the setup folder as follows:

        disk_as_sersic = False -> setup
        disk_as_sersic = True -> setup___disk_as_sersic
        """
        if not self.disk_as_sersic:
            return ""
        elif self.disk_as_sersic:
            return f"__{conf.instance.setup_tag.get('light', 'disk_as_sersic')}"

    @property
    def bulge_light_and_mass_profile(self):
        """
        The light and mass profile of a bulge component of a galaxy.

        By default, this is returned as an  _EllipticalSersic_ profile without a radial gradient, however
        the _SetupPipeline_ inputs can be customized to change this to include a radial gradient.
        """
        if not self.bulge_mass_to_light_ratio_gradient:
            return af.PriorModel(lmp.EllipticalSersic)
        return af.PriorModel(lmp.EllipticalSersicRadialGradient)

    @property
    def disk_light_and_mass_profile(self):
        """
        The light and mass profile of a disk component of a galaxy.

        By default, this is returned as an  _EllipticalExponential_ profile without a radial gradient, however
        the _SetupPipeline_ inputs can be customized to change this to an _EllipticalSersic_ or to include a radial
        gradient.
        """

        if self.disk_as_sersic:
            if not self.disk_mass_to_light_ratio_gradient:
                return af.PriorModel(lmp.EllipticalSersic)
            return af.PriorModel(lmp.EllipticalSersicRadialGradient)
        else:
            if not self.disk_mass_to_light_ratio_gradient:
                return af.PriorModel(lmp.EllipticalExponential)
            return af.PriorModel(lmp.EllipticalExponentialRadialGradient)

    def set_mass_to_light_ratios_of_light_and_mass_profiles(
        self, light_and_mass_profiles
    ):
        """
        For an input list of _LightMassProfile_'s which will represent a galaxy with a light-dark mass model, set all
        the mass-to-light ratios of every light and mass profile to the same value if a constant mass-to-light ratio
        is being used, else keep them as free parameters.

        Parameters
        ----------
        light_and_mass_profiles : [LightMassProfile]
            The light and mass profiles which have their mass-to-light ratios changed.
        """

        if self.constant_mass_to_light_ratio:

            for profile in light_and_mass_profiles[1:]:
                profile.mass_to_light_ratio = light_and_mass_profiles[
                    0
                ].mass_to_light_ratio


class AbstractSetupSource:
    pass


class SetupSourceSersic(AbstractSetupSource):
    def __init__(self):
        pass

    @property
    def model_type(self):
        return "sersic"

    @property
    def tag(self):
        """Generate a tag of the parameetric source model.
        """
        return f"{conf.instance.setup_tag.get('source', 'source')}[{self.model_type}]"


class SetupSourceInversion(AbstractSetupSource):
    def __init__(
        self,
        pixelization: pix.Pixelization = None,
        regularization: reg.Regularization = None,
        inversion_pixels_fixed: float = None,
    ):
        """The setup of the source modeling of a pipeline, which controls how PyAutoGalaxy template pipelines runs,
        for example controlling the _Pixelization_ and _Regularization_ used by a source model which uses an
        _Inversion_.

        Users can write their own pipelines which do not use or require the *SetupSource* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        pixelization : pix.Pixelization or None
           If the pipeline uses an _Inversion_ to reconstruct the galaxy's light, this determines the
           *Pixelization* used.
        regularization : reg.Regularization or None
           If the pipeline uses an _Inversion_ to reconstruct the galaxy's light, this determines the
           *Regularization* scheme used.
        inversion_pixels_fixed : float
            The fixed number of source pixels used by a _Pixelization_ class that takes as input a fixed number of
            pixels.
        """

        self._pixelization = pixelization
        self.regularization = regularization

        self.inversion_pixels_fixed = inversion_pixels_fixed

    @property
    def model_type(self):
        """Generate a tag if an _Inversion_ is used to  *Pixelization* used to reconstruct the galaxy's light, which
        is the sum of the pixelization and regularization tags.
        """
        if self._pixelization is None or self.regularization is None:
            return ""

        return f"{self.pixelization_tag}{self.inversion_pixels_fixed_tag}{self.regularization_tag}"

    @property
    def tag(self):
        """Generate a tag if an _Inversion_ is used to  *Pixelization* used to reconstruct the galaxy's light, which
        is the sum of the pixelization and regularization tags.
        """
        if self._pixelization is None or self.regularization is None:
            return ""

        return f"{conf.instance.setup_tag.get('source', 'source')}[{self.model_type}]"

    @property
    def pixelization(self):
        """
        The _PriorModel_ used to set up the pixelization model in pipelines. This _PriorModel_ has its number of
        pixels fixed to a certain value if the *inversion_pixels_fixed* parameter is input.
        """
        if (
            self._pixelization is not pix.VoronoiBrightnessImage
            or self.inversion_pixels_fixed is None
        ):
            return self._pixelization

        pixelization = af.PriorModel(self._pixelization)
        pixelization.pixels = self.inversion_pixels_fixed
        return pixelization

    @property
    def inversion_pixels_fixed_tag(self):
        """Generate a tag if an _Inversion_ is used to  *Pixelization* used to reconstruct the galaxy's light, which
        is the sum of the pixelization and regularization tags.
        """
        if self.inversion_pixels_fixed is None:
            return ""

        if self._pixelization is not pix.VoronoiBrightnessImage:
            return ""

        return f"_{str(self.inversion_pixels_fixed)}"

    @property
    def pixelization_tag(self):
        """Tag the *Pixelization* scheme used by the pipeline, if an inversion is usse to reconstruct the galaxy's
        light.

        The pixelization tag is loaded from the config file 'autogalaxy_workspace/config/label.ini' in the [tag]
        seciton.

        For the default regulariation tags, this changes the setup folder as follows:

        pixelization = None -> setup
        pixelization = pix.Rectangular -> setup__pix_rect
        pixelization = pix.VoronoiMagnification -> setup__pix_voro_mag
        """
        if self._pixelization is None:
            return ""
        else:
            return (
                f"{conf.instance.setup_tag.get('source', 'pixelization')}_"
                f"{conf.instance.setup_tag.get('pixelization', self._pixelization().__class__.__name__)}"
            )

    @property
    def regularization_tag(self):
        """Tag the *Regularization* scheme used by the pipeline, if an inversion is usse to reconstruct the galaxy's
        light.

        The regularization tag is loaded from the config file 'autogalaxy_workspace/config/label.ini' in the [tag]
        seciton.

        For the default regulariation tags, this changes the setup folder as follows:

        regularization = None -> setup
        regularization = reg.Constant -> setup__reg_const
        regularization = reg.AdaptiveBrightness -> setup__reg_adapt_bright
        """
        if self.regularization is None:
            return ""
        else:
            return (
                f"__{conf.instance.setup_tag.get('source', 'regularization')}_"
                f"{conf.instance.setup_tag.get('regularization', self.regularization().__class__.__name__)}"
            )


class SetupSMBH:
    def __init__(self, include_smbh: bool = False, smbh_centre_fixed: bool = True):
        """The setup of a super massive black hole (SMBH) in the mass model of a PyAutoGalaxy template pipeline run..

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        include_smbh : bool
            If True, a super-massive black hole (SMBH) is included in the mass model as a _PointMass_.
        smbh_centre_fixed : bool
            If True, the super-massive black hole's centre is fixed to a value input by the pipeline, else it is
            free to vary in the model.
        """
        self.include_smbh = include_smbh
        self.smbh_centre_fixed = smbh_centre_fixed

    @property
    def tag(self):
        return f"{conf.instance.setup_tag.get('smbh', 'smbh')}[{self.smbh_centre_tag}]"

    @property
    def smbh_centre_tag(self):
        """Generate a tag if the lens mass model includes a _PointMass_ representing a super-massive black hole (smbh).

        The tag includes whether the _PointMass_ centre is fixed or fitted for as a free parameter.

        This changes the setup folder as follows:

        include_smbh = False -> setup
        include_smbh = True, smbh_centre_fixed=True -> setup___smbh_centre_fixed
        include_smbh = True, smbh_centre_fixed=False -> setup___smbh_centre_free
        """
        if not self.include_smbh:
            return ""

        if self.smbh_centre_fixed:

            smbh_centre_tag = conf.instance.setup_tag.get("smbh", "smbh_centre_fixed")

        else:

            smbh_centre_tag = conf.instance.setup_tag.get("smbh", "smbh_centre_free")

        return f"{smbh_centre_tag}"

    def smbh_from_centre(self, centre, centre_sigma=0.1):
        """
        Create a _PriorModel_ of a _PointMass_ _MassProfile_ if *include_smbh* is True, which is fitted for in the
        mass-model too represent a super-massive black-hole (smbh).

        The centre of the smbh is an input parameter of the functiono, and this centre is either fixed to the input
        values as an instance or fitted for as a model.

        Parameters
        ----------
        centre : (float, float)
            The centre of the _PointMass_ that repreents the super-massive black hole.
        centre_fixed : bool
            If True, the centre is fixed to the input values, else it is fitted for as free parameters.
        centre_sigma : float
            If the centre is free, this is the sigma value of each centre's _GaussianPrior_.
        """
        if not self.include_smbh:
            return None

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
        folders: [str] = None,
        redshift_source: float = 1.0,
        setup_hyper: SetupHyper = None,
        setup_light: AbstractSetupLight = None,
        setup_mass: AbstractSetupMass = None,
        setup_source: AbstractSetupSource = None,
        setup_smbh: SetupSMBH = None,
    ):
        """The setup of a pipeline, which controls how PyAutoGalaxy template pipelines runs, for example controlling
        assumptions about the bulge-disk model or the model used to fit the source galaxy.

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
        scripts to tag the output path of the results depending on the setup parameters. This allows one to fit
        different models to a dataset in a structured path format.

        Parameters
        ----------
        folders : [str] or None
            A list of folders that the output of the pipeline are output into before the pipeline name, tags and
            phase folders.
        redshift_source : float
            The redshift of the source galaxy used by the pipeline for converting arc-seconds to kpc, masses to solMass,
            etc.
        setup_hyper : SetupHyper
            The settings of the hyper analysis if used (e.g. hyper-galaxy noise scaling).
        setup_source : AbstractSetupSource
            The settings of the source analysis (e.g. the _Pixelization and _Regularization used).
        setup_light : AbstractSetupLight
            The settings of the light profile modeling (e.g. for bulge-disk models if they are geometrically aligned).
        setup_mass : AbstractSetupMass
            The settings of the mass modeling (e.g. if a constant mass to light ratio is used).
        """

        self.folders = folders
        self.redshift_source = redshift_source
        self.setup_hyper = setup_hyper
        self.setup_source = setup_source
        self.setup_light = setup_light
        self.setup_mass = setup_mass
        self.setup_smbh = setup_smbh

        if (
            isinstance(self.setup_light, SetupLightBulgeDisk)
            and self.setup_mass is not None
        ):
            self.setup_mass.disk_as_sersic = self.setup_light.disk_as_sersic

    @property
    def tag(self):
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to.
        """

        setup_tag = conf.instance.setup_tag.get("pipeline", "pipeline")
        hyper_tag = f"__{self.setup_hyper.tag}" if self.setup_hyper is not None else ""
        source_tag = (
            f"__{self.setup_source.tag}" if self.setup_source is not None else ""
        )
        light_tag = f"__{self.setup_light.tag}" if self.setup_light is not None else ""
        mass_tag = f"__{self.setup_mass.tag}" if self.setup_mass is not None else ""
        smbh_tag = f"__{self.setup_smbh.tag}" if self.setup_smbh is not None else ""

        return f"{setup_tag}{hyper_tag}{light_tag}{mass_tag}{source_tag}{smbh_tag}"
