import pytest

import autofit as af
import autogalaxy as ag


def test__mesh_list_from_model():

    galaxies = af.Collection(galaxy=af.Model(ag.Galaxy, redshift=0.5))

    mesh_list = ag.util.model.mesh_list_from(model=af.Collection(galaxies=galaxies))

    assert mesh_list == []

    pixelization_0 = ag.Pixelization(mesh=ag.mesh.Rectangular())

    pixelization_1 = ag.Pixelization(mesh=ag.mesh.VoronoiMagnification())

    galaxies = af.Collection(
        galaxy=ag.Galaxy(
            redshift=0.5, pixelization_0=pixelization_0, pixelization_1=pixelization_1
        )
    )

    mesh_list = ag.util.model.mesh_list_from(model=af.Collection(galaxies=galaxies))

    assert isinstance(mesh_list[0], ag.mesh.Rectangular)
    assert isinstance(mesh_list[1], ag.mesh.VoronoiMagnification)

    pixelization_0 = af.Model(ag.Pixelization, mesh=ag.mesh.Rectangular)

    pixelization_1 = af.Model(ag.Pixelization, mesh=ag.mesh.VoronoiMagnification)

    galaxies = af.Collection(
        galaxy=af.Model(
            ag.Galaxy,
            redshift=0.5,
            pixelization_0=pixelization_0,
            pixelization_1=pixelization_1,
        )
    )

    mesh_list = ag.util.model.mesh_list_from(model=af.Collection(galaxies=galaxies))

    assert isinstance(mesh_list[0], ag.mesh.Rectangular)
    assert isinstance(mesh_list[1], ag.mesh.VoronoiMagnification)


def test__set_upper_limit_of_pixelization_pixels_prior():

    mesh = af.Model(ag.mesh.DelaunayBrightnessImage)

    mesh.pixels = af.UniformPrior(lower_limit=5.0, upper_limit=10.0)

    pixelization = ag.Pixelization(mesh=mesh)

    galaxies = af.Collection(source=ag.Galaxy(redshift=0.5, pixelization=pixelization))

    model = af.Collection(galaxies=galaxies)

    ag.util.model.set_upper_limit_of_pixelization_pixels_prior(
        model=model, pixels_in_mask=12
    )

    assert model.galaxies.source.pixelization.mesh.pixels.lower_limit == pytest.approx(
        5, 1.0e-4
    )
    assert model.galaxies.source.pixelization.mesh.pixels.upper_limit == pytest.approx(
        10, 1.0e-4
    )

    mesh = af.Model(ag.mesh.DelaunayBrightnessImage)
    mesh.pixels = af.UniformPrior(lower_limit=5.0, upper_limit=10.0)
    pixelization = af.Model(ag.Pixelization, mesh=mesh)
    galaxies = af.Collection(source=ag.Galaxy(redshift=0.5, pixelization=pixelization))
    model = af.Collection(galaxies=galaxies)

    ag.util.model.set_upper_limit_of_pixelization_pixels_prior(
        model=model, pixels_in_mask=12
    )

    ag.util.model.set_upper_limit_of_pixelization_pixels_prior(
        model=model, pixels_in_mask=8
    )

    assert model.galaxies.source.pixelization.mesh.pixels.lower_limit == pytest.approx(
        5, 1.0e-4
    )
    assert model.galaxies.source.pixelization.mesh.pixels.upper_limit == pytest.approx(
        8, 1.0e-4
    )

    ag.util.model.set_upper_limit_of_pixelization_pixels_prior(
        model=model, pixels_in_mask=3
    )

    assert model.galaxies.source.pixelization.mesh.pixels.lower_limit == pytest.approx(
        -7, 1.0e-4
    )
    assert model.galaxies.source.pixelization.mesh.pixels.upper_limit == pytest.approx(
        3, 1.0e-4
    )

    mesh_0 = af.Model(ag.mesh.VoronoiBrightnessImage)
    mesh_0.pixels = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    pixelization_0 = ag.Pixelization(mesh=mesh_0)

    mesh_1 = af.Model(ag.mesh.DelaunayBrightnessImage)
    mesh_1.pixels = af.UniformPrior(lower_limit=0.0, upper_limit=4.0)
    pixelization_1 = ag.Pixelization(mesh=mesh_1)

    galaxies = af.Collection(
        source=ag.Galaxy(
            redshift=0.5, pixelization_0=pixelization_0, pixelization_1=pixelization_1
        )
    )

    model = af.Collection(galaxies=galaxies)

    ag.util.model.set_upper_limit_of_pixelization_pixels_prior(
        model=model, pixels_in_mask=6
    )

    assert (
        model.galaxies.source.pixelization_0.mesh.pixels.upper_limit
        == pytest.approx(6, 1.0e-4)
    )

    assert (
        model.galaxies.source.pixelization_1.mesh.pixels.upper_limit
        == pytest.approx(4, 1.0e-4)
    )


def test__hyper_model_noise_from():

    pixelization = af.Model(
        ag.Pixelization, mesh=ag.mesh.Rectangular, regularization=ag.reg.Constant
    )

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
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

    assert model.galaxies.galaxy.pixelization.mesh.cls is ag.mesh.Rectangular
    assert model.galaxies.galaxy.pixelization.regularization.cls is ag.reg.Constant

    assert model.galaxies.galaxy.pixelization.prior_count == 0
    assert model.galaxies.galaxy.pixelization.regularization.prior_count == 0

    assert model.galaxies.galaxy_1.bulge.intensity == pytest.approx(1.0, 1.0e-4)

    assert isinstance(model.hyper_image_sky, af.Model)
    assert isinstance(model.hyper_background_noise, af.Model)

    assert model.hyper_image_sky.cls == ag.hyper_data.HyperImageSky
    assert model.hyper_background_noise.cls == ag.hyper_data.HyperBackgroundNoise

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
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
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
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

    pixelization = af.Model(ag.Pixelization, mesh=ag.mesh.Rectangular)

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = ag.m.MockResult(instance=instance)

    model = ag.util.model.hyper_inversion_model_from(
        setup_hyper=ag.SetupHyper(), result=result
    )

    assert isinstance(model.galaxies.galaxy.pixelization.mesh, af.Model)

    assert model.galaxies.galaxy.pixelization.mesh.cls is ag.mesh.Rectangular
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

    assert isinstance(model.galaxies.galaxy.pixelization.mesh, af.Model)
    assert isinstance(model.hyper_image_sky, af.Model)

    assert model.hyper_background_noise is None

    assert model.hyper_image_sky.cls == ag.hyper_data.HyperImageSky

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = ag.m.MockResult(instance=instance)

    model = ag.util.model.hyper_inversion_model_from(
        result=result, setup_hyper=ag.SetupHyper()
    )

    assert model == None


def test__hyper_model_inversion_from__adds_hyper_galaxies():

    pixelization = af.Model(ag.Pixelization, mesh=ag.mesh.Rectangular)

    model = af.Collection(
        galaxies=af.Collection(
            galaxy_0=af.Model(ag.Galaxy, redshift=0.5),
            galaxy_1=af.Model(
                ag.Galaxy,
                redshift=1.0,
                bulge=ag.lp.Sersic,
                pixelization=pixelization,
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

    pixelization = af.Model(
        ag.Pixelization,
        mesh=ag.mesh.VoronoiBrightnessImage(),
        regularization=ag.reg.AdaptiveBrightness(),
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                ag.Galaxy,
                redshift=0.5,
                light=ag.lp.SersicSph(),
                mass=ag.mp.SphIsothermal(),
            ),
            source=af.Model(ag.Galaxy, redshift=1.0, pixelization=pixelization),
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
    assert isinstance(model.galaxies.source.pixelization.mesh.pixels, int)
    assert isinstance(
        model.galaxies.source.pixelization.regularization.inner_coefficient, float
    )

    model = ag.util.model.stochastic_model_from(result=result, include_lens_light=True)

    assert isinstance(model.galaxies.lens.mass.centre, af.TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, af.LogUniformPrior)
    assert isinstance(model.galaxies.source.pixelization.mesh.pixels, int)
    assert isinstance(
        model.galaxies.source.pixelization.regularization.inner_coefficient, float
    )

    model = ag.util.model.stochastic_model_from(
        result=result, include_pixelization=True
    )

    assert isinstance(model.galaxies.lens.mass.centre, af.TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, float)
    assert isinstance(model.galaxies.source.pixelization.mesh.pixels, af.UniformPrior)
    assert not isinstance(
        model.galaxies.source.pixelization.regularization.inner_coefficient,
        af.UniformPrior,
    )

    model = ag.util.model.stochastic_model_from(
        result=result, include_regularization=True
    )

    assert isinstance(model.galaxies.lens.mass.centre, af.TuplePrior)
    assert isinstance(model.galaxies.lens.light.intensity, float)
    assert isinstance(model.galaxies.source.pixelization.mesh.pixels, int)
    assert isinstance(
        model.galaxies.source.pixelization.regularization.inner_coefficient,
        af.UniformPrior,
    )
