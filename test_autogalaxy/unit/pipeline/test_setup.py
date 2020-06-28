import autofit as af
import autogalaxy as ag
from autogalaxy import exc

import pytest


def test__inversion_evidence_tolerance():

    setup = ag.PipelineSetup(pixelization=None, inversion_evidence_tolerance=None)
    assert setup.inversion_evidence_tolerance == -1.0

    setup = ag.PipelineSetup(
        pixelization=ag.pix.Rectangular(), inversion_evidence_tolerance=None
    )
    assert setup.inversion_evidence_tolerance == -1.0

    setup = ag.PipelineSetup(
        pixelization=ag.pix.Rectangular(), inversion_evidence_tolerance=1.0
    )
    assert setup.inversion_evidence_tolerance == 1.0


def test__hyper_searches():

    setup = ag.PipelineSetup(hyper_galaxies=False)
    assert setup.hyper_galaxies_search == None

    setup = ag.PipelineSetup(hyper_galaxies=True)
    assert setup.hyper_galaxies_search.n_live_points == 75
    assert setup.hyper_galaxies_search.evidence_tolerance == pytest.approx(
        0.084, 1.0e-4
    )

    setup = ag.PipelineSetup(
        hyper_galaxies=True, hyper_galaxies_search=af.DynestyStatic(n_live_points=51)
    )
    assert setup.hyper_galaxies_search.n_live_points == 51
    assert setup.hyper_galaxies_search.evidence_tolerance == pytest.approx(0.06, 1.0e-4)

    setup = ag.PipelineSetup(inversion_search=None)
    assert setup.inversion_search.n_live_points == 50
    assert setup.inversion_search.evidence_tolerance == pytest.approx(0.059, 1.0e-4)

    setup = ag.PipelineSetup(inversion_search=af.DynestyStatic(n_live_points=51))
    assert setup.inversion_search.n_live_points == 51

    setup = ag.PipelineSetup(hyper_combined_search=af.DynestyStatic(n_live_points=51))
    assert setup.hyper_combined_search.n_live_points == 51

    setup = ag.PipelineSetup(
        hyper_galaxies=True,
        inversion_evidence_tolerance=0.5,
        pixelization=ag.pix.Rectangular(),
    )
    assert setup.hyper_galaxies_search.evidence_tolerance == 0.5
    assert setup.inversion_search.evidence_tolerance == 0.5
    assert setup.hyper_combined_search.evidence_tolerance == 0.5

    with pytest.raises(exc.PipelineException):

        ag.PipelineSetup(
            hyper_galaxies=True,
            hyper_galaxies_search=af.DynestyStatic(n_live_points=51),
            inversion_evidence_tolerance=0.5,
        )

    with pytest.raises(exc.PipelineException):
        ag.PipelineSetup(
            inversion_search=af.DynestyStatic(n_live_points=51),
            inversion_evidence_tolerance=3.0,
        )

    with pytest.raises(exc.PipelineException):
        ag.PipelineSetup(
            hyper_combined_search=af.DynestyStatic(n_live_points=51),
            inversion_evidence_tolerance=3.0,
        )


def test__hyper_galaxies_tag():

    setup = ag.PipelineSetup(hyper_galaxies=False)
    assert setup.hyper_galaxies_tag == ""

    setup = ag.PipelineSetup(hyper_galaxies=True)
    assert setup.hyper_galaxies_tag == "_galaxies"


def test__hyper_image_sky_tag():
    setup = ag.PipelineSetup(hyper_image_sky=False)
    assert setup.hyper_galaxies_tag == ""

    setup = ag.PipelineSetup(hyper_image_sky=True)
    assert setup.hyper_image_sky_tag == "_bg_sky"


def test__hyper_background_noise_tag():
    setup = ag.PipelineSetup(hyper_background_noise=False)
    assert setup.hyper_galaxies_tag == ""

    setup = ag.PipelineSetup(hyper_background_noise=True)
    assert setup.hyper_background_noise_tag == "_bg_noise"


def test__hyper_tag():

    setup = ag.PipelineSetup(
        hyper_galaxies=True, hyper_image_sky=True, hyper_background_noise=True
    )

    assert setup.hyper_tag == "__hyper_galaxies_bg_sky_bg_noise"

    setup = ag.PipelineSetup(hyper_galaxies=True, hyper_background_noise=True)

    assert setup.hyper_tag == "__hyper_galaxies_bg_noise"


def test__pixelization_tag():
    setup = ag.PipelineSetup(pixelization=None)
    assert setup.pixelization_tag == ""
    setup = ag.PipelineSetup(pixelization=ag.pix.Rectangular)
    assert setup.pixelization_tag == "pix_rect"
    setup = ag.PipelineSetup(pixelization=ag.pix.VoronoiBrightnessImage)
    assert setup.pixelization_tag == "pix_voro_image"


def test__regularization_tag():
    setup = ag.PipelineSetup(regularization=None)
    assert setup.regularization_tag == ""
    setup = ag.PipelineSetup(regularization=ag.reg.Constant)
    assert setup.regularization_tag == "__reg_const"
    setup = ag.PipelineSetup(regularization=ag.reg.AdaptiveBrightness)
    assert setup.regularization_tag == "__reg_adapt_bright"


def test__inversion_tag():
    setup = ag.PipelineSetup(pixelization=None)
    assert setup.inversion_tag == ""
    setup = ag.PipelineSetup(regularization=None)
    assert setup.inversion_tag == ""
    setup = ag.PipelineSetup(
        pixelization=ag.pix.Rectangular, regularization=ag.reg.Constant
    )
    assert setup.inversion_tag == "__pix_rect__reg_const"
    setup = ag.PipelineSetup(
        pixelization=ag.pix.VoronoiBrightnessImage,
        regularization=ag.reg.AdaptiveBrightness,
    )
    assert setup.inversion_tag == "__pix_voro_image__reg_adapt_bright"


def test__light_centre_tag():

    setup = ag.PipelineSetup(light_centre=None)
    assert setup.light_centre_tag == ""
    setup = ag.PipelineSetup(light_centre=(2.0, 2.0))
    assert setup.light_centre_tag == "__light_centre_(2.00,2.00)"
    setup = ag.PipelineSetup(light_centre=(3.0, 4.0))
    assert setup.light_centre_tag == "__light_centre_(3.00,4.00)"
    setup = ag.PipelineSetup(light_centre=(3.027, 4.033))
    assert setup.light_centre_tag == "__light_centre_(3.03,4.03)"


def test__align_bulge_disk_tags():

    light = ag.PipelineSetup(align_bulge_disk_centre=False)
    assert light.align_bulge_disk_centre_tag == ""
    light = ag.PipelineSetup(align_bulge_disk_centre=True)
    assert light.align_bulge_disk_centre_tag == "_centre"

    light = ag.PipelineSetup(align_bulge_disk_elliptical_comps=False)
    assert light.align_bulge_disk_elliptical_comps_tag == ""
    light = ag.PipelineSetup(align_bulge_disk_elliptical_comps=True)
    assert light.align_bulge_disk_elliptical_comps_tag == "_ell"


def test__bulge_disk_tag():
    light = ag.PipelineSetup(
        align_bulge_disk_centre=False, align_bulge_disk_elliptical_comps=False
    )
    assert light.align_bulge_disk_tag == ""

    light = ag.PipelineSetup(
        align_bulge_disk_centre=True, align_bulge_disk_elliptical_comps=False
    )
    print(light.align_bulge_disk_tag)
    assert light.align_bulge_disk_tag == "__align_bulge_disk_centre"

    light = ag.PipelineSetup(
        align_bulge_disk_centre=True, align_bulge_disk_elliptical_comps=True
    )
    assert light.align_bulge_disk_tag == "__align_bulge_disk_centre_ell"


def test__disk_as_sersic_tag():
    light = ag.PipelineSetup(disk_as_sersic=False)
    assert light.disk_as_sersic_tag == ""
    light = ag.PipelineSetup(disk_as_sersic=True)
    assert light.disk_as_sersic_tag == "__disk_sersic"


def test__number_of_gaussians_tag():
    setup = ag.PipelineSetup()
    assert setup.number_of_gaussians_tag == ""
    setup = ag.PipelineSetup(number_of_gaussians=1)
    assert setup.number_of_gaussians_tag == "__gaussians_x1"
    setup = ag.PipelineSetup(number_of_gaussians=2)
    assert setup.number_of_gaussians_tag == "__gaussians_x2"


def test__tag():

    setup = ag.PipelineSetup(
        pixelization=ag.pix.Rectangular,
        regularization=ag.reg.Constant,
        light_centre=(1.0, 2.0),
    )

    setup.type_tag = setup.inversion_tag

    assert setup.tag == "setup__pix_rect__reg_const__light_centre_(1.00,2.00)"

    setup = ag.PipelineSetup(number_of_gaussians=1)

    assert setup.tag == "setup__gaussians_x1"
