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
        galaxy=ag.Galaxy(
            redshift=0.5, pixelization=ag.pix.Rectangular, regularization=ag.reg.Constant
        )
    )

    pixelization = ag.util.model.pixelization_from_model(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert type(pixelization) == type(ag.pix.Rectangular)

    pixelization = af.PriorModel(ag.pix.VoronoiBrightnessImage)
    pixelization.pixels = 100

    galaxies = af.CollectionPriorModel(
        galaxy=ag.Galaxy(
            redshift=0.5, pixelization=pixelization, regularization=ag.reg.Constant
        )
    )

    pixelization = ag.util.model.pixelization_from_model(
        model=af.CollectionPriorModel(galaxies=galaxies)
    )

    assert type(pixelization) == type(ag.pix.Rectangular)


def test__has_pixelization():

    source_galaxy = ag.Galaxy(redshift=0.5)

    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
    )

    galaxies = af.CollectionPriorModel(galaxy=ag.GalaxyModel(redshift=0.5))

    assert phase_imaging_7x7.has_pixelization is False

    source_galaxy = ag.Galaxy(
        redshift=0.5,
        pixelization=ag.pix.Rectangular(),
        regularization=ag.reg.Constant(),
    )

    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
    )

    assert phase_imaging_7x7.has_pixelization is True

    source_galaxy = ag.GalaxyModel(
        redshift=0.5, pixelization=ag.pix.Rectangular, regularization=ag.reg.Constant
    )

    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
    )

    assert phase_imaging_7x7.has_pixelization is True

    pixelization = af.PriorModel(ag.pix.VoronoiBrightnessImage)
    pixelization.pixels = 100

    source_galaxy = ag.GalaxyModel(
        redshift=0.5, pixelization=pixelization, regularization=ag.reg.Constant
    )

    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
    )

    assert phase_imaging_7x7.has_pixelization is True


def test__pixelization_is_model():

    source_galaxy = ag.Galaxy(redshift=0.5)

    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
    )

    galaxies = af.CollectionPriorModel(galaxy=ag.GalaxyModel(redshift=0.5))

    assert phase_imaging_7x7.pixelization_is_model == False

    source_galaxy = ag.Galaxy(
        redshift=0.5,
        pixelization=ag.pix.Rectangular(),
        regularization=ag.reg.Constant(),
    )

    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
    )

    assert phase_imaging_7x7.pixelization_is_model == False

    source_galaxy = ag.GalaxyModel(
        redshift=0.5, pixelization=ag.pix.Rectangular, regularization=ag.reg.Constant
    )

    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
    )

    assert phase_imaging_7x7.pixelization_is_model == True

    pixelization = af.PriorModel(ag.pix.VoronoiBrightnessImage)
    pixelization.pixels = 100

    source_galaxy = ag.GalaxyModel(
        redshift=0.5, pixelization=pixelization, regularization=ag.reg.Constant
    )

    phase_imaging_7x7 = ag.PhaseImaging(
        galaxies=[source_galaxy], search=mock.MockSearch(name="test_phase")
    )

    assert phase_imaging_7x7.pixelization_is_model == True
