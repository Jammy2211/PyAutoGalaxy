from os import path

import autofit as af
import autogalaxy as ag
import numpy as np
import pytest
from astropy import cosmology as cosmo
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestHyperMethods:
    def test__associate_images(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.galaxy = ag.Galaxy(redshift=0.5)
        galaxies.source = ag.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        hyper_galaxy_image_path_dict = {
            ("galaxies", "galaxy"): ag.Array2D.ones(
                shape_native=(3, 3), pixel_scales=1.0
            ),
            ("galaxies", "source"): ag.Array2D.full(
                fill_value=2.0, shape_native=(3, 3), pixel_scales=1.0
            ),
        }

        results = mock.MockResults(
            instance=instance,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=ag.Array2D.full(
                fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
            ),
            use_as_hyper_dataset=True,
        )

        analysis = ag.PhaseImaging.Analysis(
            masked_imaging=masked_imaging_7x7,
            settings=ag.SettingsPhaseImaging(),
            results=results,
            cosmology=cosmo.Planck15,
        )

        instance = analysis.associate_hyper_images(instance=instance)

        assert instance.galaxies.galaxy.hyper_galaxy_image.native == pytest.approx(
            np.ones((3, 3)), 1.0e-4
        )
        assert instance.galaxies.source.hyper_galaxy_image.native == pytest.approx(
            2.0 * np.ones((3, 3)), 1.0e-4
        )

        assert instance.galaxies.galaxy.hyper_model_image.native == pytest.approx(
            3.0 * np.ones((3, 3)), 1.0e-4
        )
        assert instance.galaxies.source.hyper_model_image.native == pytest.approx(
            3.0 * np.ones((3, 3)), 1.0e-4
        )
