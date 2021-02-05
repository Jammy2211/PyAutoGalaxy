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


class TestMakeAnalysis:
    def test__masks_image_and_noise_map_correctly(
        self, phase_imaging_7x7, imaging_7x7, mask_7x7
    ):
        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert (
            analysis.masked_imaging.image.native
            == imaging_7x7.image.native * np.invert(mask_7x7)
        ).all()
        assert (
            analysis.masked_imaging.noise_map.native
            == imaging_7x7.noise_map.native * np.invert(mask_7x7)
        ).all()

    def test___phase_info_is_made(self, phase_imaging_7x7, imaging_7x7, mask_7x7):
        phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        file_phase_info = path.join(
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

    def test__masked_imaging__settings_inputs_are_used_in_masked_imaging(
        self, imaging_7x7, mask_7x7
    ):
        phase_imaging_7x7 = ag.PhaseImaging(
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(
                    grid_class=ag.Grid2D,
                    grid_inversion_class=ag.Grid2D,
                    sub_size=3,
                    signal_to_noise_limit=1.0,
                    bin_up_factor=2,
                    psf_shape_2d=(3, 3),
                )
            ),
            search=mock.MockSearch("phase_imaging_7x7"),
        )

        assert phase_imaging_7x7.settings.settings_masked_imaging.sub_size == 3
        assert (
            phase_imaging_7x7.settings.settings_masked_imaging.signal_to_noise_limit
            == 1.0
        )
        assert phase_imaging_7x7.settings.settings_masked_imaging.bin_up_factor == 2
        assert phase_imaging_7x7.settings.settings_masked_imaging.psf_shape_2d == (3, 3)

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(analysis.masked_dataset.grid, ag.Grid2D)
        assert isinstance(analysis.masked_dataset.grid_inversion, ag.Grid2D)

        phase_imaging_7x7 = ag.PhaseImaging(
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(
                    grid_class=ag.Grid2DIterate,
                    sub_size=3,
                    fractional_accuracy=0.99,
                    sub_steps=[2],
                )
            ),
            search=mock.MockSearch("phase_imaging_7x7"),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=mock.MockResults()
        )

        assert isinstance(analysis.masked_dataset.grid, ag.Grid2DIterate)
        assert analysis.masked_dataset.grid.sub_size == 1
        assert analysis.masked_dataset.grid.fractional_accuracy == 0.99
        assert analysis.masked_dataset.grid.sub_steps == [2]

    def test__masked_imaging__uses_signal_to_noise_limit(
        self, imaging_7x7, mask_7x7_1_pix
    ):
        imaging_snr_limit = imaging_7x7.signal_to_noise_limited_from(
            signal_to_noise_limit=1.0
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(
                    signal_to_noise_limit=1.0
                )
            ),
            search=mock.MockSearch("phase_imaging_7x7"),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7_1_pix, results=mock.MockResults()
        )
        assert (
            analysis.masked_dataset.image.native
            == imaging_snr_limit.image.native * np.invert(mask_7x7_1_pix)
        ).all()
        assert (
            analysis.masked_dataset.noise_map.native
            == imaging_snr_limit.noise_map.native * np.invert(mask_7x7_1_pix)
        ).all()

    def test__masked_imaging__uses_bin_up_factor(self, imaging_7x7, mask_7x7_1_pix):
        binned_up_imaging = imaging_7x7.binned_up_from(bin_up_factor=2)

        binned_up_mask = mask_7x7_1_pix.binned_mask_from_bin_up_factor(bin_up_factor=2)

        phase_imaging_7x7 = ag.PhaseImaging(
            settings=ag.SettingsPhaseImaging(
                settings_masked_imaging=ag.SettingsMaskedImaging(bin_up_factor=2)
            ),
            search=mock.MockSearch("phase_imaging_7x7"),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7_1_pix, results=mock.MockResults()
        )
        assert (
            analysis.masked_dataset.image.native
            == binned_up_imaging.image.native * np.invert(binned_up_mask)
        ).all()

        assert (
            analysis.masked_dataset.psf == (1.0 / 9.0) * binned_up_imaging.psf
        ).all()
        assert (
            analysis.masked_dataset.noise_map.native
            == binned_up_imaging.noise_map.native * np.invert(binned_up_mask)
        ).all()

        assert (analysis.masked_dataset.mask == binned_up_mask).all()


class TestHyperMethods:
    def test__phase_can_receive_hyper_image_and_noise_maps(self):
        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=ag.Redshift),
                galaxy1=ag.GalaxyModel(redshift=ag.Redshift),
            ),
            hyper_image_sky=ag.hyper_data.HyperImageSky,
            hyper_background_noise=ag.hyper_data.HyperBackgroundNoise,
            search=mock.MockSearch("test_phase"),
        )

        instance = phase_imaging_7x7.model.instance_from_vector([0.1, 0.2, 0.3, 0.4])

        assert instance.galaxies[0].redshift == 0.1
        assert instance.galaxies[1].redshift == 0.2
        assert instance.hyper_image_sky.sky_scale == 0.3
        assert instance.hyper_background_noise.noise_scale == 0.4

    def test__phase_is_extended_with_hyper_phases__sets_up_hyper_dataset_from_results(
        self, imaging_7x7, mask_7x7
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

        results = mock.MockResults(
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=ag.Array2D.full(
                fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
            ),
            mask=mask_7x7,
            use_as_hyper_dataset=True,
        )

        phase_imaging_7x7 = ag.PhaseImaging(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5, hyper_galaxy=ag.HyperGalaxy)
            ),
            search=mock.MockSearch("test_phase"),
        )

        analysis = phase_imaging_7x7.make_analysis(
            dataset=imaging_7x7, mask=mask_7x7, results=results
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
