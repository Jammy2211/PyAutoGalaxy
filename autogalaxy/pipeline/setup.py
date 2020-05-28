from autoconf import conf
from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg


class Setup:
    def __init__(self, general=None, light=None):

        self.general = general
        self.light = light

    def set_light_type(self, light_type):

        self.light.type_tag = light_type


class General:
    def __init__(
        self, hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
    ):

        self.hyper_galaxies = hyper_galaxies
        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

    @property
    def tag(self):
        return "general" + self.hyper_tag

    @property
    def hyper_tag(self):

        if not any(
            [self.hyper_galaxies, self.hyper_image_sky, self.hyper_background_noise]
        ):
            return ""

        return (
            "__hyper"
            + self.hyper_galaxies_tag
            + self.hyper_image_sky_tag
            + self.hyper_background_noise_tag
        )

    @property
    def hyper_galaxies_tag(self):
        """Generate a tag for if hyper-galaxies are used in a hyper_galaxies pipeline to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___hyper_galaxies
        """
        if not self.hyper_galaxies:
            return ""
        elif self.hyper_galaxies:
            return "_galaxies"

    @property
    def hyper_image_sky_tag(self):
        """Generate a tag for if the sky-background is hyper as a hyper_galaxies-parameter in a hyper_galaxies pipeline to
        customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___hyper_bg_sky
        """
        if not self.hyper_image_sky:
            return ""
        elif self.hyper_image_sky:
            return "_bg_sky"

    @property
    def hyper_background_noise_tag(self):
        """Generate a tag for if the background noise is hyper as a hyper_galaxies-parameter in a hyper_galaxies pipeline to
        customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___hyper_bg_noise
        """
        if not self.hyper_background_noise:
            return ""
        elif self.hyper_background_noise:
            return "_bg_noise"


class Light:
    def __init__(
        self,
        pixelization=pix.VoronoiBrightnessImage,
        regularization=reg.AdaptiveBrightness,
        light_centre=None,
        align_bulge_disk_centre=False,
        align_bulge_disk_phi=False,
        align_bulge_disk_axis_ratio=False,
        disk_as_sersic=False,
        number_of_gaussians=None,
    ):

        self.pixelization = pixelization
        self.regularization = regularization
        self.light_centre = light_centre
        self.align_bulge_disk_centre = align_bulge_disk_centre
        self.align_bulge_disk_phi = align_bulge_disk_phi
        self.align_bulge_disk_axis_ratio = align_bulge_disk_axis_ratio
        self.disk_as_sersic = disk_as_sersic
        self.number_of_gaussians = number_of_gaussians
        self.type_tag = None

    @property
    def tag(self):
        if self.number_of_gaussians is None:
            return (
                "light__"
                + self.type_tag
                + self.align_bulge_disk_tag
                + self.disk_as_sersic_tag
            )
        else:
            return "light__" + self.type_tag + self.number_of_gaussians_tag

    @property
    def inversion_tag(self):
        return self.pixelization_tag + self.regularization_tag

    @property
    def pixelization_tag(self):

        if self.pixelization is None:
            return ""
        else:
            return "pix_" + conf.instance.label.get(
                "tag", self.pixelization().__class__.__name__, str
            )

    @property
    def regularization_tag(self):

        if self.regularization is None:
            return ""
        else:
            return "__reg_" + conf.instance.label.get(
                "tag", self.regularization().__class__.__name__, str
            )

    @property
    def light_centre_tag(self):
        """Generate a tag for if the lens light of the pipeline and / or phase are fixed to a previous estimate, or varied \
         during he analysis, to customize phase names.

        This changes the phase name 'pipeline_name__' as follows:

        fix_lens_light = False -> pipeline_name__
        fix_lens_light = True -> pipeline_name___fix_lens_light
        """
        if self.light_centre is None:
            return ""
        else:
            y = "{0:.2f}".format(self.light_centre[0])
            x = "{0:.2f}".format(self.light_centre[1])
            return "__light_centre_(" + y + "," + x + ")"

    @property
    def align_bulge_disk_centre_tag(self):
        """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
        based on the bulge-disk model. This changee the phase name 'pipeline_name__' as follows:

        bd_align_centres = False -> pipeline_name__
        bd_align_centres = True -> pipeline_name___bd_align_centres
        """
        if not self.align_bulge_disk_centre:
            return ""
        elif self.align_bulge_disk_centre:
            return "_centre"

    @property
    def align_bulge_disk_axis_ratio_tag(self):
        """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
        based on the bulge-disk model. This changes the phase name 'pipeline_name__' as follows:

        bd_align_axis_ratio = False -> pipeline_name__
        bd_align_axis_ratio = True -> pipeline_name___bd_align_axis_ratio
        """
        if not self.align_bulge_disk_axis_ratio:
            return ""
        elif self.align_bulge_disk_axis_ratio:
            return "_axis_ratio"

    @property
    def align_bulge_disk_phi_tag(self):
        """Generate a tag for if the bulge and disk of a bulge-disk system are aligned or not, to customize phase names \
        based on the bulge-disk model. This changes the phase name 'pipeline_name__' as follows:

        bd_align_phi = False -> pipeline_name__
        bd_align_phi = True -> pipeline_name___bd_align_phi
        """
        if not self.align_bulge_disk_phi:
            return ""
        elif self.align_bulge_disk_phi:
            return "_phi"

    @property
    def align_bulge_disk_tag(self):
        """Generate a tag for the alignment of the geometry of the bulge and disk of a bulge-disk system, to customize \
        phase names based on the bulge-disk model. This adds together the bulge_disk tags generated in the 3 functions
        above
        """

        if not any(
            [
                self.align_bulge_disk_centre,
                self.align_bulge_disk_axis_ratio,
                self.align_bulge_disk_phi,
            ]
        ):
            return ""

        return (
            "__align_bulge_disk"
            + self.align_bulge_disk_centre_tag
            + self.align_bulge_disk_axis_ratio_tag
            + self.align_bulge_disk_phi_tag
        )

    @property
    def disk_as_sersic_tag(self):
        """Generate a tag for if the disk component of a bulge-disk light profile fit of the pipeline is modeled as a \
        Sersic or the default profile of an Exponential.

        This changes the phase name 'pipeline_name__' as follows:

        disk_as_sersic = False -> pipeline_name__
        disk_as_sersic = True -> pipeline_name___disk_as_sersic
        """
        if not self.disk_as_sersic:
            return "__disk_exp"
        elif self.disk_as_sersic:
            return "__disk_sersic"

    @property
    def number_of_gaussians_tag(self):
        if self.number_of_gaussians is None:
            return ""
        else:
            return "__gaussians_x" + str(self.number_of_gaussians)
