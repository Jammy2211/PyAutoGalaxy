import autogalaxy as ag


class TestPipelineGeneralSettings:
    def test__hyper_galaxies_tag(self):

        general = ag.setup.General(hyper_galaxies=False)
        assert general.hyper_galaxies_tag == ""

        general = ag.setup.General(hyper_galaxies=True)
        assert general.hyper_galaxies_tag == "_galaxies"

    def test__hyper_image_sky_tag(self):
        general = ag.setup.General(hyper_image_sky=False)
        assert general.hyper_galaxies_tag == ""

        general = ag.setup.General(hyper_image_sky=True)
        assert general.hyper_image_sky_tag == "_bg_sky"

    def test__hyper_background_noise_tag(self):
        general = ag.setup.General(hyper_background_noise=False)
        assert general.hyper_galaxies_tag == ""

        general = ag.setup.General(hyper_background_noise=True)
        assert general.hyper_background_noise_tag == "_bg_noise"

    def test__tag(self):

        general = ag.setup.General(
            hyper_galaxies=True, hyper_image_sky=True, hyper_background_noise=True
        )

        assert general.tag == "general__hyper_galaxies_bg_sky_bg_noise"

        general = ag.setup.General(hyper_galaxies=True, hyper_background_noise=True)

        assert general.tag == "general__hyper_galaxies_bg_noise"


class TestPipelineGalaxySettings:
    def test__pixelization_tag(self):
        galaxy = ag.setup.Light(pixelization=None)
        assert galaxy.pixelization_tag == ""
        galaxy = ag.setup.Light(pixelization=ag.pix.Rectangular)
        assert galaxy.pixelization_tag == "pix_rect"
        galaxy = ag.setup.Light(pixelization=ag.pix.VoronoiBrightnessImage)
        assert galaxy.pixelization_tag == "pix_voro_image"

    def test__regularization_tag(self):
        galaxy = ag.setup.Light(regularization=None)
        assert galaxy.regularization_tag == ""
        galaxy = ag.setup.Light(regularization=ag.reg.Constant)
        assert galaxy.regularization_tag == "__reg_const"
        galaxy = ag.setup.Light(regularization=ag.reg.AdaptiveBrightness)
        assert galaxy.regularization_tag == "__reg_adapt_bright"

    def test__light_centre_tag(self):

        galaxy = ag.setup.Light(light_centre=None)
        assert galaxy.light_centre_tag == ""
        galaxy = ag.setup.Light(light_centre=(2.0, 2.0))
        assert galaxy.light_centre_tag == "__light_centre_(2.00,2.00)"
        galaxy = ag.setup.Light(light_centre=(3.0, 4.0))
        assert galaxy.light_centre_tag == "__light_centre_(3.00,4.00)"
        galaxy = ag.setup.Light(light_centre=(3.027, 4.033))
        assert galaxy.light_centre_tag == "__light_centre_(3.03,4.03)"

    def test__align_bulge_disk_tags(self):

        light = ag.setup.Light(align_bulge_disk_centre=False)
        assert light.align_bulge_disk_centre_tag == ""
        light = ag.setup.Light(align_bulge_disk_centre=True)
        assert light.align_bulge_disk_centre_tag == "_centre"

        light = ag.setup.Light(align_bulge_disk_axis_ratio=False)
        assert light.align_bulge_disk_axis_ratio_tag == ""
        light = ag.setup.Light(align_bulge_disk_axis_ratio=True)
        assert light.align_bulge_disk_axis_ratio_tag == "_axis_ratio"

        light = ag.setup.Light(align_bulge_disk_phi=False)
        assert light.align_bulge_disk_phi_tag == ""
        light = ag.setup.Light(align_bulge_disk_phi=True)
        assert light.align_bulge_disk_phi_tag == "_phi"

    def test__bulge_disk_tag(self):
        light = ag.setup.Light(
            align_bulge_disk_centre=False,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=False,
        )
        assert light.align_bulge_disk_tag == ""

        light = ag.setup.Light(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=False,
        )
        assert light.align_bulge_disk_tag == "__align_bulge_disk_centre"

        light = ag.setup.Light(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=False,
            align_bulge_disk_phi=True,
        )
        assert light.align_bulge_disk_tag == "__align_bulge_disk_centre_phi"

        light = ag.setup.Light(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=True,
            align_bulge_disk_phi=True,
        )
        assert light.align_bulge_disk_tag == "__align_bulge_disk_centre_axis_ratio_phi"

    def test__disk_as_sersic_tag(self):
        light = ag.setup.Light(disk_as_sersic=False)
        assert light.disk_as_sersic_tag == "__disk_exp"
        light = ag.setup.Light(disk_as_sersic=True)
        assert light.disk_as_sersic_tag == "__disk_sersic"

    def test__number_of_gaussians_tag(self):
        galaxy = ag.setup.Light()
        assert galaxy.number_of_gaussians_tag == ""
        galaxy = ag.setup.Light(number_of_gaussians=1)
        assert galaxy.number_of_gaussians_tag == "__gaussians_x1"
        galaxy = ag.setup.Light(number_of_gaussians=2)
        assert galaxy.number_of_gaussians_tag == "__gaussians_x2"

    def test__tag(self):

        light = ag.setup.Light(align_bulge_disk_phi=True)
        light.type_tag = ""

        assert light.tag == "light____align_bulge_disk_phi__disk_exp"

        light = ag.setup.Light(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=True,
            disk_as_sersic=True,
        )

        light.type_tag = "lol"

        assert (
            light.tag == "light__lol__align_bulge_disk_centre_axis_ratio__disk_sersic"
        )

        light = ag.setup.Light(
            align_bulge_disk_centre=True,
            align_bulge_disk_axis_ratio=True,
            disk_as_sersic=True,
            number_of_gaussians=2,
        )
        light.type_tag = "test"

        assert light.tag == "light__test__gaussians_x2"
