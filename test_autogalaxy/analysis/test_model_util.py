import autofit as af
from autofit.mapper.prior.prior import TuplePrior
import autogalaxy as ag
from autogalaxy.mock import mock
from autogalaxy import hyper_data as hd


# def test__pixelization_from_model():
#
#     galaxies = af.CollectionPriorModel(galaxy=af.Model(ag.Galaxy, redshift=0.5))
#
#     pixelization = ag.util.model.pixelization_from(
#         model=af.CollectionPriorModel(galaxies=galaxies)
#     )
#
#     assert pixelization is None
#
#     galaxies = af.CollectionPriorModel(
#         galaxy=ag.Galaxy(
#             redshift=0.5,
#             pixelization=ag.pix.Rectangular(),
#             regularization=ag.reg.Constant(),
#         )
#     )
#
#     pixelization = ag.util.model.pixelization_from(
#         model=af.CollectionPriorModel(galaxies=galaxies)
#     )
#
#     assert isinstance(pixelization, ag.pix.Rectangular)
#
#     galaxies = af.CollectionPriorModel(
#         galaxy=af.Model(ag.Galaxy,
#             redshift=0.5,
#             pixelization=ag.pix.Rectangular,
#             regularization=ag.reg.Constant,
#         )
#     )
#
#     pixelization = ag.util.model.pixelization_from(
#         model=af.CollectionPriorModel(galaxies=galaxies)
#     )
#
#     assert type(pixelization) == type(ag.pix.Rectangular)
#
#
# def test__has_pixelization():
#
#     galaxies = af.CollectionPriorModel(galaxy=af.Model(ag.Galaxy, redshift=0.5))
#
#     assert galaxies.has_model(cls=ag.pix.Pixelization) is False
#
#     galaxies = af.CollectionPriorModel(
#         galaxy=af.Model(ag.Galaxy,
#             redshift=0.5,
#             pixelization=ag.pix.Rectangular(),
#             regularization=ag.reg.Constant(),
#         )
#     )
#
#     assert galaxies.has_instance(cls=ag.pix.Pixelization) is True
#     assert galaxies.has_model(cls=ag.pix.Pixelization) is False
#
#     galaxies = af.CollectionPriorModel(
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
#
#
# def test__pixelization_is_model():
#
#     galaxies = af.CollectionPriorModel(galaxy=af.Model(ag.Galaxy, redshift=0.5))
#
#     pixelization_is_model = ag.util.model.pixelization_is_model_from(
#         model=af.CollectionPriorModel(galaxies=galaxies)
#     )
#
#     assert pixelization_is_model == False
#
#     galaxies = af.CollectionPriorModel(
#         galaxy=ag.Galaxy(
#             redshift=0.5,
#             pixelization=ag.pix.Rectangular(),
#             regularization=ag.reg.Constant(),
#         )
#     )
#
#     pixelization_is_model = ag.util.model.pixelization_is_model_from(
#         model=af.CollectionPriorModel(galaxies=galaxies)
#     )
#
#     assert pixelization_is_model == False
#
#     galaxies = af.CollectionPriorModel(
#         galaxy=af.Model(ag.Galaxy,
#             redshift=0.5,
#             pixelization=ag.pix.Rectangular,
#             regularization=ag.reg.Constant,
#         )
#     )
#
#     pixelization_is_model = ag.util.model.pixelization_is_model_from(
#         model=af.CollectionPriorModel(galaxies=galaxies)
#     )
#
#     assert pixelization_is_model == True


def test__hyper_model_from():

    model = af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(
            galaxy=af.Model(
                ag.Galaxy,
                redshift=0.5,
                pixelization=ag.pix.Rectangular,
                regularization=ag.reg.Constant,
            ),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.EllipticalSersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = mock.MockResult(instance=instance)

    model = ag.util.model.hyper_model_from(setup_hyper=ag.SetupHyper(), result=result)

    assert isinstance(model.galaxies.galaxy.pixelization, af.PriorModel)
    assert isinstance(model.galaxies.galaxy.regularization, af.PriorModel)

    assert model.galaxies.galaxy.pixelization.cls is ag.pix.Rectangular
    assert model.galaxies.galaxy.regularization.cls is ag.reg.Constant
    assert model.galaxies.galaxy_1.bulge.intensity == 1.0

    assert model.hyper_image_sky is None
    assert model.hyper_background_noise is None

    model = ag.util.model.hyper_model_from(result=result, setup_hyper=None)

    assert model == None

    model = ag.util.model.hyper_model_from(
        setup_hyper=ag.SetupHyper(
            hyper_image_sky=ag.hyper_data.HyperImageSky,
            hyper_background_noise=ag.hyper_data.HyperBackgroundNoise,
        ),
        result=result,
        include_hyper_image_sky=True,
    )

    assert isinstance(model.galaxies.galaxy.pixelization, af.PriorModel)
    assert isinstance(model.galaxies.galaxy.regularization, af.PriorModel)
    assert isinstance(model.hyper_image_sky, af.PriorModel)
    assert isinstance(model.hyper_background_noise, af.PriorModel)

    assert model.hyper_image_sky.cls == ag.hyper_data.HyperImageSky
    assert model.hyper_background_noise.cls == ag.hyper_data.HyperBackgroundNoise

    model = af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(
            galaxy=af.Model(ag.Galaxy, redshift=0.5),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.EllipticalSersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = mock.MockResult(instance=instance)

    model = ag.util.model.hyper_model_from(result=result, setup_hyper=ag.SetupHyper())

    assert model == None


def test__hyper_model_from__adds_hyper_galaxies():

    model = af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(
            galaxy_0=af.Model(ag.Galaxy, redshift=0.5),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.EllipticalSersic),
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

    result = mock.MockResult(
        instance=instance,
        path_galaxy_tuples=path_galaxy_tuples,
        hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
    )

    setup_hyper = ag.SetupHyper()
    setup_hyper.hyper_galaxy_names = ["galaxy_0"]

    model = ag.util.model.hyper_model_from(result=result, setup_hyper=setup_hyper)

    assert isinstance(model.galaxies.galaxy_0, af.PriorModel)
    assert model.galaxies.galaxy_0.redshift == 0.5
    assert model.galaxies.galaxy_0.hyper_galaxy.cls is ag.HyperGalaxy
    assert model.galaxies.galaxy_1.hyper_galaxy is None

    setup_hyper = ag.SetupHyper()
    setup_hyper.hyper_galaxy_names = ["galaxy_0", "galaxy_1"]

    model = ag.util.model.hyper_model_from(result=result, setup_hyper=setup_hyper)

    assert isinstance(model.galaxies.galaxy_0, af.PriorModel)
    assert model.galaxies.galaxy_0.redshift == 0.5
    assert model.galaxies.galaxy_0.hyper_galaxy.cls is ag.HyperGalaxy
    assert isinstance(model.galaxies.galaxy_1, af.PriorModel)
    assert model.galaxies.galaxy_1.redshift == 1.0
    assert model.galaxies.galaxy_1.hyper_galaxy.cls is ag.HyperGalaxy


def test__stochastic_model_from():

    model = af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(
            lens=af.Model(
                ag.Galaxy,
                redshift=0.5,
                light=ag.lp.SphericalSersic(),
                mass=ag.mp.SphericalIsothermal(),
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

    model = af.CollectionPriorModel(
        galaxies=af.CollectionPriorModel(lens=af.Model(ag.Galaxy, redshift=0.5))
    )

    result = mock.MockResult(instance=instance, model=model)

    model = ag.util.model.stochastic_model_from(model=model, result=result)

    assert isinstance(model.galaxies.lens.mass.centre, TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, float)
    assert isinstance(model.galaxies.source.pixelization.pixels, int)
    assert isinstance(model.galaxies.source.regularization.inner_coefficient, float)

    model = ag.util.model.stochastic_model_from(
        model=model, result=result, include_lens_light=True
    )

    assert isinstance(model.galaxies.lens.mass.centre, TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, af.UniformPrior)
    assert isinstance(model.galaxies.source.pixelization.pixels, int)
    assert isinstance(model.galaxies.source.regularization.inner_coefficient, float)

    model = ag.util.model.stochastic_model_from(
        model=model, result=result, include_pixelization=True
    )

    assert isinstance(model.galaxies.lens.mass.centre, TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, float)
    assert isinstance(model.galaxies.source.pixelization.pixels, af.UniformPrior)
    assert not isinstance(
        model.galaxies.source.regularization.inner_coefficient, af.UniformPrior
    )

    model = ag.util.model.stochastic_model_from(
        model=model, result=result, include_regularization=True
    )

    assert isinstance(model.galaxies.lens.mass.centre, TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, float)
    assert isinstance(model.galaxies.source.pixelization.pixels, int)
    assert isinstance(
        model.galaxies.source.regularization.inner_coefficient, af.UniformPrior
    )
