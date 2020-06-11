from os import path

import autofit as af
import autogalaxy as ag
import numpy as np
import pytest
from test_autogalaxy.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestMakeAnalysis:
    def test__masks_image_and_noise_map_correctly(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        assert (
            analysis.masked_imaging.image.in_2d
            == imaging_7x7.image.in_2d * np.invert(mask_7x7)
        ).all()
        assert (
            analysis.masked_imaging.noise_map.in_2d
            == imaging_7x7.noise_map.in_2d * np.invert(mask_7x7)
        ).all()

    def test___phase_info_is_made(self, phase_imaging_7x7, imaging_7x7, mask_7x7):
        phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock_pipeline.MockResults()
        )

        file_phase_info = "{}/{}".format(
            phase_imaging_7x7.search.paths.output_path, "phase.info"
        )

        phase_info = open(file_phase_info, "r")

        search = phase_info.readline()
        sub_size = phase_info.readline()
        psf_shape_2d = phase_info.readline()
        cosmology = phase_info.readline()

        phase_info.close()

        assert search == "Optimizer = MockSearch \n"
        assert sub_size == "Sub-grid size = 2 \n"
        assert psf_shape_2d == "PSF shape = None \n"
        assert (
            cosmology
            == 'Cosmology = FlatLambdaCDM(name="Planck15", H0=67.7 km / (Mpc s), Om0=0.307, Tcmb0=2.725 K, '
            "Neff=3.05, m_nu=[0.   0.   0.06] eV, Ob0=0.0486) \n"
        )


class TestHyperMethods:
    def test__phase_is_extended_with_hyper_phases__sets_up_hyper_dataset_from_results(
        self, imaging_7x7, mask_7x7
    ):

        galaxies = af.ModelInstance()
        galaxies.galaxy = ag.Galaxy(redshift=0.5)
        galaxies.source = ag.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        hyper_galaxy_image_path_dict = {
            ("galaxies", "galaxy"): ag.Array.ones(shape_2d=(3, 3), pixel_scales=1.0),
            ("galaxies", "source"): ag.Array.full(
                fill_value=2.0, shape_2d=(3, 3), pixel_scales=1.0
            ),
        }

        results = mock_pipeline.MockResults(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=ag.Array.full(fill_value=3.0, shape_2d=(3, 3)),
            mask=mask_7x7,
            use_as_hyper_dataset=True,
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            phase_name="test_phase",
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5, hyper_galaxy=ag.HyperGalaxy)
            ),
            search=mock_pipeline.MockSearch(),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
        )

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "galaxy")].in_2d
            == np.ones((3, 3))
        ).all()

        assert (
            analysis.hyper_galaxy_image_path_dict[("galaxies", "source")].in_2d
            == 2.0 * np.ones((3, 3))
        ).all()

        assert (analysis.hyper_model_image.in_2d == 3.0 * np.ones((3, 3))).all()
