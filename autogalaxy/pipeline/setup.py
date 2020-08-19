from autoconf import conf
import autofit as af
from autoarray.inversion import pixelizations as pix
from autogalaxy import exc


class SetupPipeline:
    def __init__(
        self,
        folders=None,
        hyper_galaxies=False,
        hyper_image_sky=False,
        hyper_background_noise=False,
        hyper_galaxy_phase_first=False,
        hyper_galaxies_search=None,
        inversion_search=None,
        hyper_combined_search=None,
        pixelization=None,
        regularization=None,
        light_centre=None,
        align_bulge_disk_centre=False,
        align_bulge_disk_elliptical_comps=False,
        disk_as_sersic=False,
        number_of_gaussians=None,
        inversion_pixels_fixed=None,
        evidence_tolerance=None,
    ):
        """The setup of a pipeline, which controls how PyAutoGalaxy template pipelines runs, for example controlling
        assumptions about the bulge-disk model or the number of Gaussians used for multi-Gaussian fitting.

        Users can write their own pipelines which do not use or require the *SetupPipeline* class.

        This class enables pipeline tagging, whereby the setup of the pipeline is used in the template pipeline
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
        hyper_galaxies_search : af.NonLinearSearch or None
            The non-linear search used by every hyper-galaxies phase.
        inversion_search : af.NonLinearSearch or None
            The non-linear search used by every inversion phase.
        hyper_combined_search : af.NonLinearSearch or None
            The non-linear search used by every hyper combined phase.
        pixelization : ag.pix.Pixelization
           If the pipeline uses an *Inversion* to reconstruct the galaxy's light, this determines the
           *Pixelization* used.
        regularization : ag.reg.Regularization
           If the pipeline uses an *Inversion* to reconstruct the galaxy's light, this determines the
           *Regularization* scheme used.
        light_centre : (float, float)
           If input, a fixed (y,x) centre of the galaxy is used for the light profile model which is not treated as a
            free parameter by the non-linear search.
        align_bulge_disk_centre : bool
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will align the centre of the bulge and disk components and not fit them separately.
        align_bulge_disk_phi : bool
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will align the rotation angles phi of the bulge and disk components and not fit them separately.
        align_bulge_disk_axis_ratio : bool
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will align the axis-ratios of the bulge and disk components and not fit them separately.
        disk_as_sersic : bool
            If a bulge + disk light model (e.g. EllipticalSersic + EllipticalExponential) is used to fit the galaxy,
            *True* will use an EllipticalSersic for the disk instead of an EllipticalExponential.
        number_of_gaussians : int
            If a multi-Gaussian light model is used to fit the galaxy, this determines the number of Gaussians.
        """

        self.folders = folders

        self._pixelization = pixelization
        self.regularization = regularization

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

        self.hyper_galaxy_phase_first = hyper_galaxy_phase_first

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.light_centre = light_centre

        self.align_bulge_disk_centre = align_bulge_disk_centre
        self.align_bulge_disk_elliptical_comps = align_bulge_disk_elliptical_comps
        self.disk_as_sersic = disk_as_sersic
        self.number_of_gaussians = number_of_gaussians

        self.inversion_pixels_fixed = inversion_pixels_fixed

    @property
    def pixelization(self):

        if (
            self._pixelization is not pix.VoronoiBrightnessImage
            or self.inversion_pixels_fixed is None
        ):
            return self._pixelization

        pixelization = af.PriorModel(self._pixelization)
        pixelization.pixels = self.inversion_pixels_fixed
        return pixelization

    @property
    def hyper_tag(self):
        """Tag ithe hyper pipeline features used in a hyper pipeline to customize pipeline output paths.
        """
        if not any(
            [self.hyper_galaxies, self.hyper_image_sky, self.hyper_background_noise]
        ):
            return ""

        return (
            "__"
            + conf.instance.tag.get("pipeline", "hyper", str)
            + self.hyper_galaxies_tag
            + self.hyper_image_sky_tag
            + self.hyper_background_noise_tag
        )

    @property
    def hyper_galaxies_tag(self):
        """Tag if hyper-galaxies are used in a hyper pipeline to customize pipeline output paths.

        This is used to generate an overall hyper tag in *hyper_tag*.
        """
        if not self.hyper_galaxies:
            return ""
        elif self.hyper_galaxies:
            return "_" + conf.instance.tag.get("pipeline", "hyper_galaxies", str)

    @property
    def hyper_image_sky_tag(self):
        """Tag if the sky-background is hyper as a hyper_galaxies-parameter in a hyper pipeline to customize pipeline
        output paths.

        This is used to generate an overall hyper tag in *hyper_tag*.
        """
        if not self.hyper_image_sky:
            return ""
        elif self.hyper_image_sky:
            return "_" + conf.instance.tag.get("pipeline", "hyper_image_sky", str)

    @property
    def hyper_background_noise_tag(self):
        """Tag if the background noise is hyper as a hyper_galaxies-parameter in a hyper pipeline to customize pipeline
        output paths.

        This is used to generate an overall hyper tag in *hyper_tag*.
        """
        if not self.hyper_background_noise:
            return ""
        elif self.hyper_background_noise:
            return "_" + conf.instance.tag.get(
                "pipeline", "hyper_background_noise", str
            )

    @property
    def tag(self):
        """Generate the pipeline's overall tag, which customizes the 'setup' folder the results are output to.
        """
        return (
            conf.instance.tag.get("pipeline", "pipeline", str)
            + self.hyper_tag
            + self.inversion_tag
            + self.light_centre_tag
            + self.align_bulge_disk_tag
            + self.disk_as_sersic_tag
            + self.number_of_gaussians_tag
        )

    @property
    def inversion_tag(self):
        """Generate a tag if an *Inversion* is used to  *Pixelization* used to reconstruct the galaxy's light, which 
        is the sum of the pixelization and regularization tags.
        """
        if self._pixelization is None or self.regularization is None:
            return ""

        return (
            "__"
            + self.pixelization_tag
            + self.inversion_pixels_fixed_tag
            + self.regularization_tag
        )

    @property
    def inversion_tag_no_underscore(self):
        """Generate a tag if an *Inversion* is used to  *Pixelization* used to reconstruct the galaxy's light, which
        is the sum of the pixelization and regularization tags.
        """
        if self._pixelization is None or self.regularization is None:
            return ""

        return (
            self.pixelization_tag
            + self.inversion_pixels_fixed_tag
            + self.regularization_tag
        )

    @property
    def inversion_pixels_fixed_tag(self):
        """Generate a tag if an *Inversion* is used to  *Pixelization* used to reconstruct the galaxy's light, which
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
                conf.instance.tag.get("pipeline", "pixelization", str)
                + "_"
                + conf.instance.tag.get(
                    "pixelization", self._pixelization().__class__.__name__, str
                )
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
                "__"
                + conf.instance.tag.get("pipeline", "regularization", str)
                + "_"
                + conf.instance.tag.get(
                    "regularization", self.regularization().__class__.__name__, str
                )
            )

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
            return (
                "__"
                + conf.instance.tag.get("pipeline", "light_centre", str)
                + "_("
                + y
                + ","
                + x
                + ")"
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
            return "_" + conf.instance.tag.get(
                "pipeline", "align_bulge_disk_centre", str
            )

    @property
    def align_bulge_disk_elliptical_comps_tag(self):
        """Tag if the ellipticity of the bulge and disk of a bulge-disk system are aligned or not, to customize pipeline
        output paths based on the bulge-disk model. 
        
        This is used to generate an overall align-bulge disk tag in *align_bulge_disk_tag*.
        """
        if not self.align_bulge_disk_elliptical_comps:
            return ""
        elif self.align_bulge_disk_elliptical_comps:
            return "_" + conf.instance.tag.get(
                "pipeline", "align_bulge_disk_elliptical_comps", str
            )

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

        return (
            "__"
            + conf.instance.tag.get("pipeline", "align_bulge_disk", str)
            + self.align_bulge_disk_centre_tag
            + self.align_bulge_disk_elliptical_comps_tag
        )

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
            return "__" + conf.instance.tag.get("pipeline", "disk_as_sersic", str)

    @property
    def number_of_gaussians_tag(self):
        """Tag the number of Gaussians if modeling the galaxy using multiple Gaussians light profiles.

        This changes the setup folder as follows:

        number_of_gaussians = None -> setup
        number_of_gaussians = 2 = True -> setup__gaussians_x2
        number_of_gaussians = 4 = True -> setup__gaussians_x4
        """
        if self.number_of_gaussians is None:
            return ""
        else:
            return (
                "__"
                + conf.instance.tag.get("pipeline", "number_of_gaussians", str)
                + "_x"
                + str(self.number_of_gaussians)
            )
