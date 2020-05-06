import autogalaxy as ag
import numpy as np
from astropy import cosmology as cosmo

import autofit as af
from test_autolens.mock import mock_pipeline


class TestImagePassing:
    def test___image_dict(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.lens = ag.Galaxy(redshift=0.5)
        galaxies.source = ag.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = ag.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            cosmology=cosmo.Planck15,
            image_path="files/",
            results=mock_pipeline.MockResults(),
        )

        result = ag.PhaseImaging.Result(
            samples=mock_pipeline.MockSamples(max_log_likelihood_instance=instance),
            previous_model=af.ModelMapper(),
            analysis=analysis,
            optimizer=None,
        )

        image_dict = result.image_galaxy_dict
        assert isinstance(image_dict[("galaxies", "lens")], np.ndarray)
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

        result.instance.galaxies.lens = ag.Galaxy(redshift=0.5)

        image_dict = result.image_galaxy_dict
        assert (image_dict[("galaxies", "lens")].in_2d == np.zeros((7, 7))).all()
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)
