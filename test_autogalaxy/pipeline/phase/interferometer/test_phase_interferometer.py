from os import path

import numpy as np
import pytest

import autofit as af
import autogalaxy as ag
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestHyperMethods:
    def test__phase_is_extended_with_hyper_phases__sets_up_hyper_images(
        self, interferometer_7, mask_7x7
    ):
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

        hyper_galaxy_visibilities_path_dict = {
            ("galaxies", "galaxy"): ag.Visibilities.full(
                fill_value=4.0, shape_slim=(7,)
            ),
            ("galaxies", "source"): ag.Visibilities.full(
                fill_value=5.0, shape_slim=(7,)
            ),
        }

        results = mock.MockResults(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=ag.Array2D.full(
                fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
            ),
            hyper_galaxy_visibilities_path_dict=hyper_galaxy_visibilities_path_dict,
            hyper_model_visibilities=ag.Visibilities.full(
                fill_value=6.0, shape_slim=(7,)
            ),
            mask=mask_7x7,
            use_as_hyper_dataset=True,
        )

        phase_interferometer_7 = ag.PhaseInterferometer(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5, hyper_galaxy=ag.HyperGalaxy)
            ),
            search=mock.MockSearch(),
            real_space_mask=mask_7x7,
        )

        setup_hyper = ag.SetupHyper(
            hyper_search_with_inversion=mock.MockSearch("test_phase")
        )

        phase_interferometer_7.extend_with_hyper_phase(setup_hyper=setup_hyper)

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7, mask=mask_7x7, results=results
        )

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "galaxy")].native
            == np.ones((3, 3))
        ).all()

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "source")].native
            == 2.0 * np.ones((3, 3))
        ).all()

        assert (analysis.hyper_model_image.native == 3.0 * np.ones((3, 3))).all()

        assert (
            analysis.hyper_galaxy_visibilities_path_dict[("galaxies", "galaxy")]
            == 4.0 + 4.0j * np.ones((7,))
        ).all()

        assert (
            analysis.hyper_galaxy_visibilities_path_dict[("galaxies", "source")]
            == 5.0 + 5.0j * np.ones((7,))
        ).all()

        assert (analysis.hyper_model_visibilities == 6.0 + 6.0j * np.ones((7,))).all()
