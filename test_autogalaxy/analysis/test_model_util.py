import autofit as af
from autofit.mapper.prior.prior import TuplePrior
import autogalaxy as ag
from autogalaxy.mock import mock
from autogalaxy import hyper_data as hd


def test__pixelization_from_model():

    galaxies = af.CollectionPriorModel(galaxy=ag.GalaxyModel(redshift=0.5))

    pixelization = ag.util.model.pixelization_from(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert pixelization is None

    galaxies = af.CollectionPriorModel(
        galaxy=ag.Galaxy(
            redshift=0.5,
            pixelization=ag.pix.Rectangular(),
            regularization=ag.reg.Constant(),
        )
    )

    pixelization = ag.util.model.pixelization_from(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert isinstance(pixelization, ag.pix.Rectangular)

    galaxies = af.CollectionPriorModel(
        galaxy=ag.GalaxyModel(
            redshift=0.5,
            pixelization=ag.pix.Rectangular,
            regularization=ag.reg.Constant,
        )
    )

    pixelization = ag.util.model.pixelization_from(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert type(pixelization) == type(ag.pix.Rectangular)


def test__has_pixelization():

    galaxies = af.CollectionPriorModel(galaxy=ag.GalaxyModel(redshift=0.5))

    has_pixelization = ag.util.model.has_pixelization_from_model(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert has_pixelization is False

    galaxies = af.CollectionPriorModel(
        galaxy=ag.Galaxy(
            redshift=0.5,
            pixelization=ag.pix.Rectangular(),
            regularization=ag.reg.Constant(),
        )
    )

    has_pixelization = ag.util.model.has_pixelization_from_model(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert has_pixelization is True

    galaxies = af.CollectionPriorModel(
        galaxy=ag.Galaxy(
            redshift=0.5,
            pixelization=ag.pix.Rectangular,
            regularization=ag.reg.Constant,
        )
    )

    has_pixelization = ag.util.model.has_pixelization_from_model(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert has_pixelization is True


def test__pixelization_is_model():

    galaxies = af.CollectionPriorModel(galaxy=ag.GalaxyModel(redshift=0.5))

    pixelization_is_model = ag.util.model.pixelization_is_model_from(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert pixelization_is_model == False

    galaxies = af.CollectionPriorModel(
        galaxy=ag.Galaxy(
            redshift=0.5,
            pixelization=ag.pix.Rectangular(),
            regularization=ag.reg.Constant(),
        )
    )

    pixelization_is_model = ag.util.model.pixelization_is_model_from(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert pixelization_is_model == False

    galaxies = af.CollectionPriorModel(
        galaxy=ag.GalaxyModel(
            redshift=0.5,
            pixelization=ag.pix.Rectangular,
            regularization=ag.reg.Constant,
        )
    )

    pixelization_is_model = ag.util.model.pixelization_is_model_from(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert pixelization_is_model == True


# def test__make_hyper_model_from():
#
#     instance = af.ModelInstance()
#
#     instance.galaxy = ag.Galaxy(
#         redshift=0.5,
#         pixelization=ag.pix.Rectangular(),
#         regularization=ag.reg.Constant(),
#     )
#     instance.galaxy_1 = ag.Galaxy(
#         redshift=1.0, bulge=ag.lp.EllipticalSersic(intensity=10.0)
#     )
#
#     result = mock.MockResult(instance=instance)
#
#     model = ag.util.model.make_hyper_model_from(result=result)
#
#     assert isinstance(model.galaxy.pixelization, af.PriorModel)
#     assert isinstance(model.galaxy.regularization, af.PriorModel)
#
#     assert model.galaxy.pixelization.cls is ag.pix.Rectangular
#     assert model.galaxy.regularization.cls is ag.reg.Constant
#     assert model.galaxy_1.bulge.intensity == 10.0
#
#     assert model.hyper_image_sky is None
#     assert model.hyper_background_noise is None
#
#     instance = af.ModelInstance()
#
#     result = mock.MockResult(instance=instance)
#
#     model = ag.util.model.make_hyper_model_from(
#         result=result,
#         hyper_image_sky=hd.HyperImageSky,
#         hyper_background_noise=hd.HyperBackgroundNoise,
#     )
#
#     assert isinstance(model.hyper_image_sky, af.PriorModel)
#     assert isinstance(model.hyper_background_noise, af.PriorModel)
#
#     assert model.hyper_image_sky.cls == ag.hyper_data.HyperImageSky
#     assert model.hyper_background_noise.cls == ag.hyper_data.HyperBackgroundNoise


def test__make_hyper_model_from__no_pixelization_or_reg_returns_none():

    instance = af.ModelInstance()

    instance.galaxies = []

    instance.galaxies.append(ag.Galaxy(redshift=0.5))
    instance.galaxies.append(
        ag.Galaxy(redshift=1.0, bulge=ag.lp.EllipticalSersic(intensity=10.0))
    )

    result = mock.MockResult(instance=instance)

    model = ag.util.model.hyper_model_from(result=result, setup_hyper=None)

    assert model == None


# def test__make_hyper_model_from__adds_hyper_galaxies():
#
#     path_galaxy_tuples = [
#         (("galaxies", "lens"), ag.Galaxy(redshift=0.5)),
#         (("galaxies", "source"), ag.Galaxy(redshift=1.0)),
#     ]
#
#     hyper_galaxy_image_path_dict = {
#         ("galaxies", "lens"): ag.Array2D.ones(shape_native=(3, 3), pixel_scales=1.0),
#         ("galaxies", "source"): ag.Array2D.full(
#             fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
#         ),
#     }
#
#     instance = af.ModelInstance()
#
#     instance.galaxies = []
#
#     instance.galaxies.append(
#         ag.Galaxy(
#             redshift=0.5,
#             pixelization=ag.pix.Rectangular(),
#             regularization=ag.reg.Constant(),
#         )
#     )
#
#     result = mock.MockResult(
#         instance=instance,
#         path_galaxy_tuples=path_galaxy_tuples,
#         hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
#     )
#
#     model = ag.util.model.make_hyper_model_from(
#         result=result, hyper_galaxy_names="lens"
#     )
#
#     assert model.galaxies.lens.cls is ag.HyperGalaxy

#
# def test__extend_with_stochastic_phase__sets_up_model_correctly():
#
#     instance = af.ModelInstance()
#     instance.galaxies = []
#
#     instance.galaxies.append(
#         ag.Galaxy(
#             redshift=0.5,
#             light=ag.lp.SphericalSersic(),
#             mass=ag.mp.SphericalIsothermal(),
#         )
#     )
#     instance.galaxies.append(
#         ag.Galaxy(
#             redshift=1.0,
#             pixelization=ag.pix.VoronoiBrightnessImage(),
#             regularization=ag.reg.AdaptiveBrightness(),
#         )
#     )
#
#     result = mock.MockResult(instance=instance)
#
#     model = af.CollectionPriorModel(
#         galaxies=af.CollectionPriorModel(lens=ag.GalaxyModel(redshift=0.5))
#     )
#
#     model = ag.util.model.make_stochastic_model_from(model=model, result=result)
#
#     assert isinstance(model.lens.mass.centre, TuplePrior)
#     assert isinstance(model.lens.light.intensity, float)
#     assert isinstance(model.source.pixelization.pixels, int)
#     assert isinstance(model.source.regularization.inner_coefficient, float)
#
#     stop
#
#     phase_extended = phase.extend_with_stochastic_phase(include_lens_light=True)
#
#     model = phase_extended.make_model(instance=instance)
#
#     assert isinstance(model.lens.mass.centre, TuplePrior)
#     assert isinstance(model.lens.light.intensity, af.UniformPrior)
#     assert isinstance(model.source.pixelization.pixels, int)
#     assert isinstance(model.source.regularization.inner_coefficient, float)
#
#     phase_extended = phase.extend_with_stochastic_phase(include_pixelization=True)
#
#     model = phase_extended.make_model(instance=instance)
#
#     assert isinstance(model.lens.mass.centre, TuplePrior)
#     assert isinstance(model.lens.light.intensity, float)
#     assert isinstance(model.source.pixelization.pixels, af.UniformPrior)
#     assert not isinstance(
#         model.source.regularization.inner_coefficient, af.UniformPrior
#     )
#
#     phase_extended = phase.extend_with_stochastic_phase(include_regularization=True)
#
#     model = phase_extended.make_model(instance=instance)
#
#     assert isinstance(model.lens.mass.centre, TuplePrior)
#     assert isinstance(model.lens.light.intensity, float)
#     assert isinstance(model.source.pixelization.pixels, int)
#     assert isinstance(model.source.regularization.inner_coefficient, af.UniformPrior)
#
#     phase = ag.PhaseInterferometer(
#         search=mock.MockSearch(),
#         real_space_mask=mask_7x7,
#         galaxies=af.CollectionPriorModel(lens=ag.GalaxyModel(redshift=0.5)),
#     )
#
#     phase_extended = phase.extend_with_stochastic_phase()
#
#     model = phase_extended.make_model(instance=instance)
#
#     assert isinstance(model.lens.mass.centre, TuplePrior)
#     assert isinstance(model.lens.light.intensity, float)
#     assert isinstance(model.source.pixelization.pixels, int)
#     assert isinstance(model.source.regularization.inner_coefficient, float)
