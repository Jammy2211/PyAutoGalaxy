import autofit as af
import autogalaxy as ag
from autogalaxy import exc

import pytest


class TestSetupHyper:
    def test__hyper_searches(self):
        setup = ag.SetupHyper(hyper_galaxies=False)
        assert setup.hyper_galaxies_search == None

        setup = ag.SetupHyper(hyper_galaxies=True)
        assert setup.hyper_galaxies_search.n_live_points == 75
        assert setup.hyper_galaxies_search.evidence_tolerance == pytest.approx(
            0.084, 1.0e-4
        )

        setup = ag.SetupHyper(
            hyper_galaxies=True,
            hyper_galaxies_search=af.DynestyStatic(n_live_points=51),
        )
        assert setup.hyper_galaxies_search.n_live_points == 51
        assert setup.hyper_galaxies_search.evidence_tolerance == pytest.approx(
            0.06, 1.0e-4
        )

        setup = ag.SetupHyper(inversion_search=None)
        assert setup.inversion_search.n_live_points == 50
        assert setup.inversion_search.evidence_tolerance == pytest.approx(0.059, 1.0e-4)

        setup = ag.SetupHyper(inversion_search=af.DynestyStatic(n_live_points=51))
        assert setup.inversion_search.n_live_points == 51

        setup = ag.SetupHyper(hyper_combined_search=af.DynestyStatic(n_live_points=51))
        assert setup.hyper_combined_search.n_live_points == 51

        setup = ag.SetupHyper(hyper_galaxies=True, evidence_tolerance=0.5)
        assert setup.hyper_galaxies_search.evidence_tolerance == pytest.approx(
            0.084, 1.0e-4
        )
        assert setup.inversion_search.evidence_tolerance == 0.5
        assert setup.hyper_combined_search.evidence_tolerance == 0.5

        with pytest.raises(exc.PipelineException):
            ag.SetupHyper(
                hyper_galaxies=True,
                hyper_galaxies_search=af.DynestyStatic(n_live_points=51),
                evidence_tolerance=0.5,
            )

        with pytest.raises(exc.PipelineException):
            ag.SetupHyper(
                inversion_search=af.DynestyStatic(n_live_points=51),
                evidence_tolerance=3.0,
            )

        with pytest.raises(exc.PipelineException):
            ag.SetupHyper(
                hyper_combined_search=af.DynestyStatic(n_live_points=51),
                evidence_tolerance=3.0,
            )

    def test__hyper_galaxies_tag(self):
        setup = ag.SetupHyper(hyper_galaxies=False)
        assert setup.hyper_galaxies_tag == ""

        setup = ag.SetupHyper(hyper_galaxies=True)
        assert setup.hyper_galaxies_tag == "galaxies"

    def test__hyper_image_sky_tag(self):
        setup = ag.SetupHyper(hyper_image_sky=False)
        assert setup.hyper_galaxies_tag == ""

        setup = ag.SetupHyper(hyper_image_sky=True)
        assert setup.hyper_image_sky_tag == "_bg_sky"

    def test__hyper_background_noise_tag(self):
        setup = ag.SetupHyper(hyper_background_noise=False)
        assert setup.hyper_galaxies_tag == ""

        setup = ag.SetupHyper(hyper_background_noise=True)
        assert setup.hyper_background_noise_tag == "_bg_noise"

    def test__hyper_fixed_after_source(self):
        hyper = ag.SetupHyper(hyper_fixed_after_source=False)
        assert hyper.hyper_fixed_after_source_tag == ""

        hyper = ag.SetupHyper(hyper_fixed_after_source=True)
        assert hyper.hyper_fixed_after_source_tag == "_fixed"

    def test__hyper_tag(self):

        setup = ag.SetupHyper(
            hyper_galaxies=False, hyper_image_sky=False, hyper_background_noise=False
        )

        assert setup.tag == ""

        setup = ag.SetupHyper(
            hyper_galaxies=True, hyper_image_sky=True, hyper_background_noise=True
        )

        assert setup.tag == "hyper[galaxies_bg_sky_bg_noise]"

        setup = ag.SetupHyper(
            hyper_galaxies=True,
            hyper_background_noise=True,
            hyper_fixed_after_source=True,
        )

        assert setup.tag == "hyper[galaxies_bg_noise_fixed]"


class TestAbstractSetupLight:
    def test__align_centre_to_light_centre(self):

        light = af.PriorModel(ag.mp.SphericalIsothermal)

        source = ag.SetupLightSersic(light_centre=(1.0, 2.0))

        light = source.align_centre_to_light_centre(light=light)

        assert light.centre == (1.0, 2.0)


class TestSetupLightSersic:
    def test__light_centre_tag(self):
        setup = ag.SetupLightSersic(light_centre=None)
        assert setup.light_centre_tag == ""
        setup = ag.SetupLightSersic(light_centre=(2.0, 2.0))
        assert setup.light_centre_tag == "__light_centre_(2.00,2.00)"
        setup = ag.SetupLightSersic(light_centre=(3.0, 4.0))
        assert setup.light_centre_tag == "__light_centre_(3.00,4.00)"
        setup = ag.SetupLightSersic(light_centre=(3.027, 4.033))
        assert setup.light_centre_tag == "__light_centre_(3.03,4.03)"

    def test__tag_and_type(self):
        setup = ag.SetupLightSersic(light_centre=None)
        assert setup.tag == "light[bulge]"
        setup = ag.SetupLightSersic(light_centre=(3.027, 4.033))
        assert setup.tag == "light[bulge__light_centre_(3.03,4.03)]"


class TestSetupLightBulgeDisk:
    def test__align_bulge_disk_tags(self):
        light = ag.SetupLightBulgeDisk(align_bulge_disk_centre=False)
        assert light.align_bulge_disk_centre_tag == ""
        light = ag.SetupLightBulgeDisk(align_bulge_disk_centre=True)
        assert light.align_bulge_disk_centre_tag == "_centre"

        light = ag.SetupLightBulgeDisk(align_bulge_disk_elliptical_comps=False)
        assert light.align_bulge_disk_elliptical_comps_tag == ""
        light = ag.SetupLightBulgeDisk(align_bulge_disk_elliptical_comps=True)
        assert light.align_bulge_disk_elliptical_comps_tag == "_ell"

    def test__bulge_disk_tag(self):
        light = ag.SetupLightBulgeDisk(
            align_bulge_disk_centre=False, align_bulge_disk_elliptical_comps=False
        )
        assert light.align_bulge_disk_tag == ""

        light = ag.SetupLightBulgeDisk(
            align_bulge_disk_centre=True, align_bulge_disk_elliptical_comps=False
        )

        assert light.align_bulge_disk_tag == "__align_bulge_disk_centre"

        light = ag.SetupLightBulgeDisk(
            align_bulge_disk_centre=True, align_bulge_disk_elliptical_comps=True
        )
        assert light.align_bulge_disk_tag == "__align_bulge_disk_centre_ell"

    def test__disk_as_sersic_tag(self):
        light = ag.SetupLightBulgeDisk(disk_as_sersic=False)
        assert light.disk_as_sersic_tag == ""
        light = ag.SetupLightBulgeDisk(disk_as_sersic=True)
        assert light.disk_as_sersic_tag == "__disk_sersic"

    def test__tag_and_type(self):

        light = ag.SetupLightBulgeDisk()
        assert light.tag == "light[bulge_disk]"

        light = ag.SetupLightBulgeDisk(
            align_bulge_disk_centre=True,
            align_bulge_disk_elliptical_comps=True,
            disk_as_sersic=True,
        )
        assert (
            light.tag == "light[bulge_disk__align_bulge_disk_centre_ell__disk_sersic]"
        )


class TestAbstractSetupMass:
    def test__align_centre_to_mass_centre(self):

        mass = af.PriorModel(ag.mp.SphericalIsothermal)

        source = ag.SetupMassTotal(mass_centre=(1.0, 2.0))

        mass = source.align_centre_to_mass_centre(mass=mass)

        assert mass.centre == (1.0, 2.0)

    def test__unfix_mass_centre(self):

        mass = af.PriorModel(ag.mp.SphericalIsothermal)
        mass.centre = (1.0, 2.0)

        setup_mass = ag.SetupMassTotal()

        mass = af.PriorModel(ag.mp.SphericalIsothermal)

        mass = setup_mass.unfix_mass_centre(mass=mass)

        # assert mass.centre.centre_0.mean == 5.0
        # assert mass.centre.centre_0.sigma == 0.05
        # assert mass.centre.centre_1.mean == 6.0
        # assert mass.centre.centre_1.sigma == 0.05


class TestSetupMassTotal:
    def test__mass_centre_tag(self):
        setup = ag.SetupMassTotal(mass_centre=None)
        assert setup.mass_centre_tag == ""
        setup = ag.SetupMassTotal(mass_centre=(2.0, 2.0))
        assert setup.mass_centre_tag == "__mass_centre_(2.00,2.00)"
        setup = ag.SetupMassTotal(mass_centre=(3.0, 4.0))
        assert setup.mass_centre_tag == "__mass_centre_(3.00,4.00)"
        setup = ag.SetupMassTotal(mass_centre=(3.027, 4.033))
        assert setup.mass_centre_tag == "__mass_centre_(3.03,4.03)"

    def test__mass_profile_tag(self):

        setup = ag.SetupMassTotal(mass_profile=None)
        assert setup.mass_profile_tag == ""

        setup = ag.SetupMassTotal(mass_profile=ag.mp.EllipticalPowerLaw)
        assert setup.mass_profile_tag == "__power_law"

        setup = ag.SetupMassTotal(mass_profile=ag.mp.EllipticalIsothermal)
        assert setup.mass_profile_tag == "__sie"

    def test__tag_and_type(self):

        setup = ag.SetupMassTotal(mass_centre=None)
        assert setup.tag == "mass[total]"
        setup = ag.SetupMassTotal(mass_centre=(3.027, 4.033))
        assert setup.tag == "mass[total__mass_centre_(3.03,4.03)]"

        setup = ag.SetupMassTotal(
            mass_centre=(3.027, 4.033), mass_profile=ag.mp.EllipticalPowerLaw
        )
        assert setup.tag == "mass[total__power_law__mass_centre_(3.03,4.03)]"

    def test__align_centre_of_mass_to_light(self):

        mass = af.PriorModel(ag.mp.SphericalIsothermal)

        source = ag.SetupMassTotal(align_light_mass_centre=False)

        mass = source.align_centre_of_mass_to_light(mass=mass, light_centre=(1.0, 2.0))

        assert mass.centre.centre_0.mean == 1.0
        assert mass.centre.centre_0.sigma == 0.1
        assert mass.centre.centre_0.mean == 1.0
        assert mass.centre.centre_0.sigma == 0.1

        source = ag.SetupMassTotal(align_light_mass_centre=True)

        mass = source.align_centre_of_mass_to_light(mass=mass, light_centre=(1.0, 2.0))

        assert mass.centre == (1.0, 2.0)


class TestSetupMassLightDark:
    def test__constant_mass_to_light_ratio_tag(self):
        setup = ag.SetupMassLightDark(constant_mass_to_light_ratio=True)
        assert setup.constant_mass_to_light_ratio_tag == "_const"
        setup = ag.SetupMassLightDark(constant_mass_to_light_ratio=False)
        assert setup.constant_mass_to_light_ratio_tag == "_free"

    def test__bulge_and_disk_mass_to_light_ratio_gradient_tag(self):
        setup = ag.SetupMassLightDark(bulge_mass_to_light_ratio_gradient=True)
        assert setup.bulge_mass_to_light_ratio_gradient_tag == "_bulge"
        setup = ag.SetupMassLightDark(bulge_mass_to_light_ratio_gradient=False)
        assert setup.bulge_mass_to_light_ratio_gradient_tag == ""

        setup = ag.SetupMassLightDark(disk_mass_to_light_ratio_gradient=True)
        assert setup.disk_mass_to_light_ratio_gradient_tag == "_disk"
        setup = ag.SetupMassLightDark(disk_mass_to_light_ratio_gradient=False)
        assert setup.disk_mass_to_light_ratio_gradient_tag == ""

    def test__mass_to_light_tag(self):
        setup = ag.SetupMassLightDark(
            constant_mass_to_light_ratio=True,
            bulge_mass_to_light_ratio_gradient=False,
            disk_mass_to_light_ratio_gradient=False,
        )
        assert setup.mass_to_light_tag == "__mlr_const"

        setup = ag.SetupMassLightDark(
            constant_mass_to_light_ratio=True,
            bulge_mass_to_light_ratio_gradient=True,
            disk_mass_to_light_ratio_gradient=False,
        )
        assert setup.mass_to_light_tag == "__mlr_const_grad_bulge"

        setup = ag.SetupMassLightDark(
            constant_mass_to_light_ratio=True,
            bulge_mass_to_light_ratio_gradient=True,
            disk_mass_to_light_ratio_gradient=True,
        )
        assert setup.mass_to_light_tag == "__mlr_const_grad_bulge_disk"

        setup = ag.SetupMassLightDark(
            constant_mass_to_light_ratio=False,
            bulge_mass_to_light_ratio_gradient=True,
            disk_mass_to_light_ratio_gradient=False,
        )
        assert setup.mass_to_light_tag == "__mlr_free_grad_bulge"

    def test__align_light_dark_tag(self):
        setup = ag.SetupMassLightDark(align_light_dark_centre=False)
        assert setup.align_light_dark_centre_tag == ""
        setup = ag.SetupMassLightDark(align_light_dark_centre=True)
        assert setup.align_light_dark_centre_tag == "__align_light_dark_centre"

    def test__align_bulge_dark_tag(self):
        setup = ag.SetupMassLightDark(align_bulge_dark_centre=False)
        assert setup.align_bulge_dark_centre_tag == ""
        setup = ag.SetupMassLightDark(align_bulge_dark_centre=True)
        assert setup.align_bulge_dark_centre_tag == "__align_bulge_dark_centre"

    def test__tag_and_type(self):

        setup = ag.SetupMassLightDark(
            constant_mass_to_light_ratio=True,
            bulge_mass_to_light_ratio_gradient=True,
            disk_mass_to_light_ratio_gradient=True,
            align_bulge_dark_centre=True,
        )
        assert (
            setup.tag
            == "mass[light_dark__mlr_const_grad_bulge_disk__align_bulge_dark_centre]"
        )

    def test__bulge_light_and_mass_profile(self):

        light = ag.SetupMassLightDark(bulge_mass_to_light_ratio_gradient=False)
        assert (
            light.bulge_light_and_mass_profile.effective_radius
            is ag.lmp.EllipticalSersic
        )

        light = ag.SetupMassLightDark(bulge_mass_to_light_ratio_gradient=True)
        assert (
            light.bulge_light_and_mass_profile.effective_radius
            is ag.lmp.EllipticalSersicRadialGradient
        )

    def test__disk_light_and_mass_profile(self):

        mass = ag.SetupMassLightDark(disk_mass_to_light_ratio_gradient=False)

        mass.disk_as_sersic = False

        assert (
            mass.disk_light_and_mass_profile.effective_radius
            is ag.lmp.EllipticalExponential
        )

        mass = ag.SetupMassLightDark(disk_mass_to_light_ratio_gradient=False)

        mass.disk_as_sersic = True

        assert (
            mass.disk_light_and_mass_profile.effective_radius is ag.lmp.EllipticalSersic
        )

        mass = ag.SetupMassLightDark(disk_mass_to_light_ratio_gradient=True)

        mass.disk_as_sersic = False

        assert (
            mass.disk_light_and_mass_profile.effective_radius
            is ag.lmp.EllipticalExponentialRadialGradient
        )

        mass = ag.SetupMassLightDark(disk_mass_to_light_ratio_gradient=True)

        mass.disk_as_sersic = True

        assert (
            mass.disk_light_and_mass_profile.effective_radius
            is ag.lmp.EllipticalSersicRadialGradient
        )

    def test__set_mass_to_light_ratios_of_light_and_mass_profiles(self):

        lmp_0 = af.PriorModel(ag.lmp.EllipticalSersic)
        lmp_1 = af.PriorModel(ag.lmp.EllipticalSersic)
        lmp_2 = af.PriorModel(ag.lmp.EllipticalSersic)

        setup = ag.SetupMassLightDark(constant_mass_to_light_ratio=False)

        setup.set_mass_to_light_ratios_of_light_and_mass_profiles(
            light_and_mass_profiles=[lmp_0, lmp_1, lmp_2]
        )

        assert lmp_0.mass_to_light_ratio != lmp_1.mass_to_light_ratio
        assert lmp_0.mass_to_light_ratio != lmp_2.mass_to_light_ratio
        assert lmp_1.mass_to_light_ratio != lmp_2.mass_to_light_ratio

        lmp_0 = af.PriorModel(ag.lmp.EllipticalSersic)
        lmp_1 = af.PriorModel(ag.lmp.EllipticalSersic)
        lmp_2 = af.PriorModel(ag.lmp.EllipticalSersic)

        setup = ag.SetupMassLightDark(constant_mass_to_light_ratio=True)

        setup.set_mass_to_light_ratios_of_light_and_mass_profiles(
            light_and_mass_profiles=[lmp_0, lmp_1, lmp_2]
        )

        assert lmp_0.mass_to_light_ratio == lmp_1.mass_to_light_ratio
        assert lmp_0.mass_to_light_ratio == lmp_2.mass_to_light_ratio
        assert lmp_1.mass_to_light_ratio == lmp_2.mass_to_light_ratio


class TestSetupSourceSersic:
    def test__tag_ang_type(self):

        setup = ag.SetupSourceSersic()

        assert setup.model_type == "sersic"
        assert setup.tag == "source[sersic]"


class TestSetupSourceInversion:
    def test__pixelization__model_depends_on_inversion_pixels_fixed(self):
        setup = ag.SetupSourceInversion()

        assert setup.pixelization is None

        setup = ag.SetupSourceInversion(pixelization=ag.pix.Rectangular)

        assert setup.pixelization is ag.pix.Rectangular

        setup = ag.SetupSourceInversion(pixelization=ag.pix.VoronoiBrightnessImage)

        assert setup.pixelization is ag.pix.VoronoiBrightnessImage

        setup = ag.SetupSourceInversion(
            pixelization=ag.pix.VoronoiBrightnessImage, inversion_pixels_fixed=100
        )

        assert isinstance(setup.pixelization, af.PriorModel)
        assert setup.pixelization.pixels == 100

    def test__pixelization_tag(self):
        setup = ag.SetupSourceInversion(pixelization=None)
        assert setup.pixelization_tag == ""
        setup = ag.SetupSourceInversion(pixelization=ag.pix.Rectangular)
        assert setup.pixelization_tag == "pix_rect"
        setup = ag.SetupSourceInversion(pixelization=ag.pix.VoronoiBrightnessImage)
        assert setup.pixelization_tag == "pix_voro_image"

    def test__regularization_tag(self):
        setup = ag.SetupSourceInversion(regularization=None)
        assert setup.regularization_tag == ""
        setup = ag.SetupSourceInversion(regularization=ag.reg.Constant)
        assert setup.regularization_tag == "__reg_const"
        setup = ag.SetupSourceInversion(regularization=ag.reg.AdaptiveBrightness)
        assert setup.regularization_tag == "__reg_adapt_bright"

    def test__inversion_pixels_fixed_tag(self):
        setup = ag.SetupSourceInversion(inversion_pixels_fixed=None)
        assert setup.inversion_pixels_fixed_tag == ""

        setup = ag.SetupSourceInversion(inversion_pixels_fixed=100)
        assert setup.inversion_pixels_fixed_tag == ""

        setup = ag.SetupSourceInversion(
            inversion_pixels_fixed=100, pixelization=ag.pix.VoronoiBrightnessImage
        )
        assert setup.inversion_pixels_fixed_tag == "_100"

    def test__tag_and_type(self):

        setup = ag.SetupSourceInversion(pixelization=None, inversion_pixels_fixed=100)
        assert setup.model_type == ""
        assert setup.tag == ""
        setup = ag.SetupSourceInversion(regularization=None, inversion_pixels_fixed=100)
        assert setup.model_type == ""
        assert setup.tag == ""
        setup = ag.SetupSourceInversion(
            pixelization=ag.pix.Rectangular,
            regularization=ag.reg.Constant,
            inversion_pixels_fixed=100,
        )
        assert setup.model_type == "pix_rect__reg_const"
        assert setup.tag == "source[pix_rect__reg_const]"
        setup = ag.SetupSourceInversion(
            pixelization=ag.pix.VoronoiBrightnessImage,
            regularization=ag.reg.AdaptiveBrightness,
            inversion_pixels_fixed=None,
        )
        assert setup.model_type == "pix_voro_image__reg_adapt_bright"
        assert setup.tag == "source[pix_voro_image__reg_adapt_bright]"

        setup = ag.SetupSourceInversion(
            pixelization=ag.pix.VoronoiBrightnessImage,
            regularization=ag.reg.AdaptiveBrightness,
            inversion_pixels_fixed=100,
        )
        assert setup.model_type == "pix_voro_image_100__reg_adapt_bright"
        assert setup.tag == "source[pix_voro_image_100__reg_adapt_bright]"


class TestSMBH:
    def test__smbh_tag(self):
        setup = ag.SetupSMBH(include_smbh=False)
        assert setup.tag == "smbh[]"

        setup = ag.SetupSMBH(include_smbh=True, smbh_centre_fixed=True)
        assert setup.tag == "smbh[centre_fixed]"

        setup = ag.SetupSMBH(include_smbh=True, smbh_centre_fixed=False)
        assert setup.tag == "smbh[centre_free]"

    def test__smbh_from_centre(self):

        setup = ag.SetupSMBH(include_smbh=False, smbh_centre_fixed=True)
        smbh = setup.smbh_from_centre(centre=(0.0, 0.0))
        assert smbh is None

        setup = ag.SetupSMBH(include_smbh=True, smbh_centre_fixed=True)
        smbh = setup.smbh_from_centre(centre=(0.0, 0.0))
        assert isinstance(smbh, af.PriorModel)
        assert smbh.centre == (0.0, 0.0)

        setup = ag.SetupSMBH(include_smbh=True, smbh_centre_fixed=False)
        smbh = setup.smbh_from_centre(centre=(0.1, 0.2), centre_sigma=0.2)
        assert isinstance(smbh, af.PriorModel)
        assert isinstance(smbh.centre[0], af.GaussianPrior)
        assert smbh.centre[0].mean == 0.1
        assert smbh.centre[0].sigma == 0.2
        assert isinstance(smbh.centre[1], af.GaussianPrior)
        assert smbh.centre[1].mean == 0.2
        assert smbh.centre[1].sigma == 0.2


class TestSetupPipeline:
    def test__tag(self):

        source = ag.SetupSourceInversion(
            pixelization=ag.pix.Rectangular, regularization=ag.reg.Constant
        )

        light = ag.SetupLightBulgeDisk(light_centre=(1.0, 2.0))

        setup = ag.SetupPipeline(setup_source=source, setup_light=light)

        setup.type_tag = setup.setup_source.tag

        assert (
            setup.tag == "setup__"
            "light[bulge_disk__light_centre_(1.00,2.00)]__"
            "source[pix_rect__reg_const]"
        )

        hyper = ag.SetupHyper(hyper_galaxies=True, hyper_background_noise=True)

        source = ag.SetupSourceInversion(
            pixelization=ag.pix.Rectangular, regularization=ag.reg.Constant
        )

        light = ag.SetupLightBulgeDisk(light_centre=(1.0, 2.0))

        setup = ag.SetupPipeline(
            setup_hyper=hyper, setup_source=source, setup_light=light
        )

        setup.type_tag = setup.setup_source.tag

        assert (
            setup.tag == "setup__hyper[galaxies_bg_noise]__"
            "light[bulge_disk__light_centre_(1.00,2.00)]__"
            "source[pix_rect__reg_const]"
        )

        smbh = ag.SetupSMBH(include_smbh=True, smbh_centre_fixed=True)

        setup = ag.SetupPipeline(setup_smbh=smbh)

        assert setup.tag == "setup__smbh[centre_fixed]"
