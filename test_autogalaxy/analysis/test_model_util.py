import pytest

import autofit as af
import autogalaxy as ag


def test__pixelization_from_model():

    galaxies = af.Collection(galaxy=af.Model(ag.Galaxy, redshift=0.5))

    pixelization = ag.util.model.pixelization_from(
        model=af.Collection(galaxies=galaxies)
    )

    assert pixelization is None

    galaxies = af.Collection(
        galaxy=ag.Galaxy(
            redshift=0.5,
            pixelization=ag.pix.Rectangular(),
            regularization=ag.reg.Constant(),
        )
    )

    pixelization = ag.util.model.pixelization_from(
        model=af.Collection(galaxies=galaxies)
    )

    assert isinstance(pixelization, ag.pix.Rectangular)

    galaxies = af.Collection(
        galaxy=af.Model(
            ag.Galaxy,
            redshift=0.5,
            pixelization=ag.pix.Rectangular,
            regularization=ag.reg.Constant,
        )
    )

    pixelization = ag.util.model.pixelization_from(
        model=af.Collection(galaxies=galaxies)
    )

    assert type(pixelization) == type(ag.pix.Rectangular)


# def test__has_pixelization():
#     galaxies = af.Collection(galaxy=af.Model(ag.Galaxy, redshift=0.5))
#
#     assert galaxies.has_model(cls=ag.pix.Pixelization) is False
#
#     galaxies = af.Collection(
#         galaxy=af.Model(
#             ag.Galaxy,
#             redshift=0.5,
#             pixelization=ag.pix.Rectangular(),
#             regularization=ag.reg.Constant(),
#         )
#     )
#
#     assert galaxies.has_instance(cls=ag.pix.Pixelization) is True
#     assert galaxies.has_model(cls=ag.pix.Pixelization) is False
#
#     galaxies = af.Collection(
#         galaxy=af.Model(
#             ag.Galaxy,
#             redshift=0.5,
#             pixelization=ag.pix.Rectangular,
#             regularization=ag.reg.Constant,
#         )
#     )
#
#     assert galaxies.has_instance(cls=ag.pix.Pixelization) is False
#     assert galaxies.has_model(cls=ag.pix.Pixelization) is True


def test__hyper_model_noise_from():
    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(
                ag.Galaxy,
                redshift=0.5,
                pixelization=ag.pix.Rectangular,
                regularization=ag.reg.Constant,
            ),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.EllSersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = ag.m.MockResult(instance=instance)

    model = ag.util.model.hyper_noise_model_from(
        setup_hyper=ag.SetupHyper(), result=result
    )

    assert model is None

    model = ag.util.model.hyper_noise_model_from(result=result, setup_hyper=None)

    assert model == None

    model = ag.util.model.hyper_noise_model_from(
        setup_hyper=ag.SetupHyper(
            hyper_image_sky=ag.hyper_data.HyperImageSky,
            hyper_background_noise=ag.hyper_data.HyperBackgroundNoise,
        ),
        result=result,
        include_hyper_image_sky=True,
    )

    assert model.galaxies.galaxy.pixelization.cls is ag.pix.Rectangular
    assert model.galaxies.galaxy.regularization.cls is ag.reg.Constant

    assert model.galaxies.galaxy.pixelization.prior_count == 0
    assert model.galaxies.galaxy.regularization.prior_count == 0

    assert model.galaxies.galaxy_1.bulge.intensity == pytest.approx(1.0, 1.0e-4)

    assert isinstance(model.hyper_image_sky, af.Model)
    assert isinstance(model.hyper_background_noise, af.Model)

    assert model.hyper_image_sky.cls == ag.hyper_data.HyperImageSky
    assert model.hyper_background_noise.cls == ag.hyper_data.HyperBackgroundNoise

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.EllSersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = ag.m.MockResult(instance=instance)

    model = ag.util.model.hyper_noise_model_from(
        result=result, setup_hyper=ag.SetupHyper()
    )

    assert model == None


def test__hyper_model_noise_from__adds_hyper_galaxies():
    model = af.Collection(
        galaxies=af.Collection(
            galaxy_0=af.Model(ag.Galaxy, redshift=0.5),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.EllSersic),
        )
    )

    instance = model.instance_from_prior_medians()

    path_galaxy_tuples = [
        (("galaxies", "galaxy_0"), ag.Galaxy(redshift=0.5)),
        (("galaxies", "galaxy_1"), ag.Galaxy(redshift=1.0)),
    ]

    hyper_galaxy_image_path_dict = {
        ("galaxies", "galaxy_0"): ag.Array2D.ones(
            shape_native=(3, 3), pixel_scales=1.0
        ),
        ("galaxies", "galaxy_1"): ag.Array2D.full(
            fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    }

    result = ag.m.MockResult(
        instance=instance,
        path_galaxy_tuples=path_galaxy_tuples,
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
    )

    setup_hyper = ag.SetupHyper()
    setup_hyper.hyper_galaxy_names = ["galaxy_0"]

    model = ag.util.model.hyper_noise_model_from(result=result, setup_hyper=setup_hyper)

    assert isinstance(model.galaxies.galaxy_0, af.Model)
    assert model.galaxies.galaxy_0.redshift == 0.5
    assert model.galaxies.galaxy_0.hyper_galaxy.cls is ag.HyperGalaxy
    assert model.galaxies.galaxy_1.hyper_galaxy is None

    setup_hyper = ag.SetupHyper()
    setup_hyper.hyper_galaxy_names = ["galaxy_0", "galaxy_1"]

    model = ag.util.model.hyper_noise_model_from(result=result, setup_hyper=setup_hyper)

    assert isinstance(model.galaxies.galaxy_0, af.Model)
    assert model.galaxies.galaxy_0.redshift == 0.5
    assert model.galaxies.galaxy_0.hyper_galaxy.cls is ag.HyperGalaxy
    assert isinstance(model.galaxies.galaxy_1, af.Model)
    assert model.galaxies.galaxy_1.redshift == 1.0
    assert model.galaxies.galaxy_1.hyper_galaxy.cls is ag.HyperGalaxy


def test__hyper_model_inversion_from():

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(
                ag.Galaxy,
                redshift=0.5,
                pixelization=ag.pix.Rectangular,
                regularization=ag.reg.Constant,
            ),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.EllSersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = ag.m.MockResult(instance=instance)

    model = ag.util.model.hyper_inversion_model_from(
        setup_hyper=ag.SetupHyper(), result=result
    )

    assert isinstance(model.galaxies.galaxy.pixelization, af.Model)
    assert isinstance(model.galaxies.galaxy.regularization, af.Model)

    assert model.galaxies.galaxy.pixelization.cls is ag.pix.Rectangular
    assert model.galaxies.galaxy.regularization.cls is ag.reg.Constant
    assert model.galaxies.galaxy_1.bulge.intensity == pytest.approx(1.0, 1.0e-4)

    assert model.hyper_image_sky is None
    assert model.hyper_background_noise is None

    model = ag.util.model.hyper_inversion_model_from(result=result, setup_hyper=None)

    assert model == None

    model = ag.util.model.hyper_inversion_model_from(
        setup_hyper=ag.SetupHyper(
            hyper_image_sky=ag.hyper_data.HyperImageSky,
            hyper_background_noise=ag.hyper_data.HyperBackgroundNoise,
        ),
        result=result,
        include_hyper_image_sky=True,
    )

    assert isinstance(model.galaxies.galaxy.pixelization, af.Model)
    assert isinstance(model.galaxies.galaxy.regularization, af.Model)
    assert isinstance(model.hyper_image_sky, af.Model)

    assert model.hyper_background_noise is None

    assert model.hyper_image_sky.cls == ag.hyper_data.HyperImageSky

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.EllSersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = ag.m.MockResult(instance=instance)

    model = ag.util.model.hyper_inversion_model_from(
        result=result, setup_hyper=ag.SetupHyper()
    )

    assert model == None


def test__hyper_model_inversion_from__adds_hyper_galaxies():
    model = af.Collection(
        galaxies=af.Collection(
            galaxy_0=af.Model(ag.Galaxy, redshift=0.5),
            galaxy_1=af.Model(
                ag.Galaxy,
                redshift=1.0,
                bulge=ag.lp.EllSersic,
                pixelization=ag.pix.Rectangular,
                regularization=ag.reg.Constant,
            ),
        )
    )

    instance = model.instance_from_prior_medians()

    path_galaxy_tuples = [
        (
            ("galaxies", "galaxy_0"),
            ag.Galaxy(redshift=0.5, hyper_galaxy=ag.HyperGalaxy(contribution_factor=1)),
        ),
        (
            ("galaxies", "galaxy_1"),
            ag.Galaxy(redshift=1.0, hyper_galaxy=ag.HyperGalaxy(contribution_factor=2)),
        ),
    ]

    hyper_galaxy_image_path_dict = {
        ("galaxies", "galaxy_0"): ag.Array2D.ones(
            shape_native=(3, 3), pixel_scales=1.0
        ),
        ("galaxies", "galaxy_1"): ag.Array2D.full(
            fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
        ),
    }

    result = ag.m.MockResult(
        instance=instance,
        path_galaxy_tuples=path_galaxy_tuples,
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
    )

    setup_hyper = ag.SetupHyper()
    setup_hyper.hyper_galaxy_names = ["galaxy_0"]

    model = ag.util.model.hyper_inversion_model_from(
        result=result, setup_hyper=setup_hyper
    )

    assert isinstance(model.galaxies.galaxy_0, af.Model)
    assert model.galaxies.galaxy_0.redshift == 0.5
    assert model.galaxies.galaxy_0.hyper_galaxy.contribution_factor == 1
    assert model.galaxies.galaxy_1.hyper_galaxy is None

    setup_hyper = ag.SetupHyper()
    setup_hyper.hyper_galaxy_names = ["galaxy_0", "galaxy_1"]

    model = ag.util.model.hyper_inversion_model_from(
        result=result, setup_hyper=setup_hyper
    )

    assert isinstance(model.galaxies.galaxy_0, af.Model)
    assert model.galaxies.galaxy_0.redshift == 0.5
    assert model.galaxies.galaxy_0.hyper_galaxy.contribution_factor == 1
    assert isinstance(model.galaxies.galaxy_1, af.Model)
    assert model.galaxies.galaxy_1.redshift == 1.0
    assert model.galaxies.galaxy_1.hyper_galaxy.contribution_factor == 2


def test__stochastic_model_from():
    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                ag.Galaxy,
                redshift=0.5,
                light=ag.lp.SphSersic(),
                mass=ag.mp.SphIsothermal(),
            ),
            source=af.Model(
                ag.Galaxy,
                redshift=1.0,
                pixelization=ag.pix.VoronoiBrightnessImage(),
                regularization=ag.reg.AdaptiveBrightness(),
            ),
        )
    )

    instance = model.instance_from_prior_medians()

    model = af.Collection(
        galaxies=af.Collection(lens=af.Model(ag.Galaxy, redshift=0.5))
    )

    result = ag.m.MockResult(instance=instance, model=model)

    model = ag.util.model.stochastic_model_from(result=result)

    assert isinstance(model.galaxies.lens.mass.centre, af.TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, float)
    assert isinstance(model.galaxies.source.pixelization.pixels, int)
    assert isinstance(model.galaxies.source.regularization.inner_coefficient, float)

    model = ag.util.model.stochastic_model_from(result=result, include_lens_light=True)

    assert isinstance(model.galaxies.lens.mass.centre, af.TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, af.LogUniformPrior)
    assert isinstance(model.galaxies.source.pixelization.pixels, int)
    assert isinstance(model.galaxies.source.regularization.inner_coefficient, float)

    model = ag.util.model.stochastic_model_from(
        result=result, include_pixelization=True
    )

    assert isinstance(model.galaxies.lens.mass.centre, af.TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, float)
    assert isinstance(model.galaxies.source.pixelization.pixels, af.UniformPrior)
    assert not isinstance(
        model.galaxies.source.regularization.inner_coefficient, af.UniformPrior
    )

    model = ag.util.model.stochastic_model_from(
        result=result, include_regularization=True
    )

    assert isinstance(model.galaxies.lens.mass.centre, af.TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, float)
    assert isinstance(model.galaxies.source.pixelization.pixels, int)
    assert isinstance(
        model.galaxies.source.regularization.inner_coefficient, af.UniformPrior
    )
