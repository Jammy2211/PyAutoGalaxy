import pytest

import autofit as af
import autogalaxy as ag


def test__hyper_model_noise_from():
    pixelization = af.Model(
        ag.Pixelization, mesh=ag.mesh.Rectangular, regularization=ag.reg.Constant
    )

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.legacy.Galaxy, redshift=0.5, pixelization=pixelization),
            galaxy_1=af.Model(ag.legacy.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = ag.m.MockResult(instance=instance)

    model = ag.util.model_legacy.hyper_noise_model_from(
        setup_adapt=ag.legacy.SetupAdapt(), result=result
    )

    assert model is None

    model = ag.util.model_legacy.hyper_noise_model_from(result=result, setup_adapt=None)

    assert model == None

    model = ag.util.model_legacy.hyper_noise_model_from(
        setup_adapt=ag.legacy.SetupAdapt(
            hyper_image_sky=ag.legacy.hyper_data.HyperImageSky,
            hyper_background_noise=ag.legacy.hyper_data.HyperBackgroundNoise,
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

    assert model.hyper_image_sky.cls == ag.legacy.hyper_data.HyperImageSky
    assert model.hyper_background_noise.cls == ag.legacy.hyper_data.HyperBackgroundNoise

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.legacy.Galaxy, redshift=0.5),
            galaxy_1=af.Model(ag.legacy.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = ag.m.MockResult(instance=instance)

    model = ag.util.model_legacy.hyper_noise_model_from(
        result=result, setup_adapt=ag.legacy.SetupAdapt()
    )

    assert model == None


def test__hyper_model_noise_from__adds_hyper_galaxies():
    model = af.Collection(
        galaxies=af.Collection(
            galaxy_0=af.Model(ag.legacy.Galaxy, redshift=0.5),
            galaxy_1=af.Model(ag.legacy.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
        )
    )

    instance = model.instance_from_prior_medians()

    path_galaxy_tuples = [
        (("galaxies", "galaxy_0"), ag.legacy.Galaxy(redshift=0.5)),
        (("galaxies", "galaxy_1"), ag.legacy.Galaxy(redshift=1.0)),
    ]

    adapt_galaxy_image_path_dict = {
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
        adapt_galaxy_image_path_dict=adapt_galaxy_image_path_dict,
    )

    setup_adapt = ag.legacy.SetupAdapt()
    setup_adapt.hyper_galaxy_names = ["galaxy_0"]

    model = ag.util.model_legacy.hyper_noise_model_from(
        result=result, setup_adapt=setup_adapt
    )

    assert isinstance(model.galaxies.galaxy_0, af.Model)
    assert model.galaxies.galaxy_0.redshift == 0.5
    assert model.galaxies.galaxy_0.hyper_galaxy.cls is ag.legacy.HyperGalaxy
    assert model.galaxies.galaxy_1.hyper_galaxy is None

    setup_adapt = ag.legacy.SetupAdapt()
    setup_adapt.hyper_galaxy_names = ["galaxy_0", "galaxy_1"]

    model = ag.util.model_legacy.hyper_noise_model_from(
        result=result, setup_adapt=setup_adapt
    )

    assert isinstance(model.galaxies.galaxy_0, af.Model)
    assert model.galaxies.galaxy_0.redshift == 0.5
    assert model.galaxies.galaxy_0.hyper_galaxy.cls is ag.legacy.HyperGalaxy
    assert isinstance(model.galaxies.galaxy_1, af.Model)
    assert model.galaxies.galaxy_1.redshift == 1.0
    assert model.galaxies.galaxy_1.hyper_galaxy.cls is ag.legacy.HyperGalaxy
