import autofit as af
import autogalaxy as ag
from autogalaxy.mock import mock


def test__pixelization_from_model():

    galaxies = af.CollectionPriorModel(galaxy=ag.GalaxyModel(redshift=0.5))

    pixelization = ag.util.model.pixelization_from_model(
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

    pixelization = ag.util.model.pixelization_from_model(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert isinstance(pixelization, ag.pix.Rectangular)

    galaxies = af.CollectionPriorModel(
        galaxy=ag.GalaxyModel(
            redshift=0.5, pixelization=ag.pix.Rectangular, regularization=ag.reg.Constant
        )
    )

    pixelization = ag.util.model.pixelization_from_model(
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
            redshift=0.5, pixelization=ag.pix.Rectangular, regularization=ag.reg.Constant
        )
    )

    has_pixelization = ag.util.model.has_pixelization_from_model(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert has_pixelization is True


def test__pixelization_is_model():

    galaxies = af.CollectionPriorModel(galaxy=ag.GalaxyModel(redshift=0.5))

    pixelization_is_model = ag.util.model.pixelization_is_model_from_model(
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

    pixelization_is_model = ag.util.model.pixelization_is_model_from_model(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert pixelization_is_model == False

    galaxies = af.CollectionPriorModel(
        galaxy=ag.GalaxyModel(
            redshift=0.5, pixelization=ag.pix.Rectangular, regularization=ag.reg.Constant
        )
    )

    pixelization_is_model = ag.util.model.pixelization_is_model_from_model(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert pixelization_is_model == True