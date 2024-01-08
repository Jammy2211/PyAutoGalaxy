import pytest

import autofit as af
import autogalaxy as ag

def test__adapt_model_from():
    class MockResult:
        def __init__(self, instance, model):
            self.instance = instance
            self.model = model

    pixelization = af.Model(ag.Pixelization, mesh=ag.mesh.Rectangular)

    model = af.Collection(
        galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization),
            galaxy_1=af.Model(ag.Galaxy, redshift=1.0, bulge=ag.lp.Sersic),
        )
    )

    instance = model.instance_from_prior_medians()

    result = MockResult(instance=instance, model=model)

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

    result = MockResult(instance=instance, model=model)
    model = ag.util.model.adapt_model_from(result=result, setup_adapt=ag.SetupAdapt())

    assert model == None
