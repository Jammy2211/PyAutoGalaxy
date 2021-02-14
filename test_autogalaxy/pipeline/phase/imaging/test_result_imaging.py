import autofit as af
import autogalaxy as ag
import numpy as np
from astropy import cosmology as cosmo
from autogalaxy.mock import mock


class TestImagePassing:
    def test___image_dict(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.galaxy = ag.Galaxy(redshift=0.5)
        galaxies.source = ag.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = ag.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            settings=ag.SettingsPhaseImaging(),
            results=mock.MockResults(),
            cosmology=cosmo.Planck15,
        )

        result = ag.PhaseImaging.Result(
            samples=mock.MockSamples(max_log_likelihood_instance=instance),
            previous_model=af.ModelMapper(),
            analysis=analysis,
            search=None,
        )

        image_dict = result.image_galaxy_dict
        assert isinstance(image_dict[("galaxies", "galaxy")], np.ndarray)
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

        result.instance.galaxies.light = ag.Galaxy(redshift=0.5)

        image_dict = result.image_galaxy_dict
        assert (image_dict[("galaxies", "galaxy")].native == np.zeros((7, 7))).all()
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)
