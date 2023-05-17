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

    # mesh_0 = af.Model(ag.mesh.VoronoiBrightnessImage)
    # mesh_0.pixels = af.UniformPrior(lower_limit=0.0, upper_limit=100.0)
    # pixelization_0 = ag.Pixelization(mesh=mesh_0)
    #
    # mesh_1 = af.Model(ag.mesh.DelaunayBrightnessImage)
    # mesh_1.pixels = af.UniformPrior(lower_limit=0.0, upper_limit=4.0)
    # pixelization_1 = ag.Pixelization(mesh=mesh_1)
    #
    # galaxies = af.Collection(
    #     source=ag.Galaxy(
    #         redshift=0.5, pixelization_0=pixelization_0, pixelization_1=pixelization_1
    #     )
    # )
    #
    # model = af.Collection(galaxies=galaxies)
    #
    # ag.util.model.set_upper_limit_of_pixelization_pixels_prior(
    #     model=model, pixels_in_mask=6
    # )
    #
    # assert (
    #     model.galaxies.source.pixelization_0.mesh.pixels.upper_limit
    #     == pytest.approx(6, 1.0e-4)
    # )
    #
    # assert (
    #     model.galaxies.source.pixelization_1.mesh.pixels.upper_limit
    #     == pytest.approx(4, 1.0e-4)
    # )


def test__adapt_model_from():
    pixelization = af.Model(ag.Pixelization, mesh=ag.mesh.Rectangular)

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = ag.m.MockResult(instance=instance)

    model = ag.util.model.adapt_model_from(setup_adapt=ag.SetupAdapt(), result=result)

    assert isinstance(model.galaxies.galaxy.pixelization.mesh, af.Model)

    assert model.galaxies.galaxy.pixelization.mesh.cls is ag.mesh.Rectangular
    assert model.galaxies.galaxy_1.bulge.intensity == pytest.approx(1.0, 1.0e-4)

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = ag.m.MockResult(instance=instance)

    model = ag.util.model.adapt_model_from(result=result, setup_adapt=ag.SetupAdapt())

    assert model == None
