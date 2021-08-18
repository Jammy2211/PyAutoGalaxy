import pytest
import numpy as np
from os import path

import autofit as af
import autogalaxy as ag

from autogalaxy.mock import mock


directory = path.dirname(path.realpath(__file__))


class TestAnalysisDataset:
    def test__modify_dataset_before_fit(self, masked_imaging_7x7):

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7,
        )

        assert analysis.dataset_modified == None

        # No hyper galaxy is model, so identical dataset is used.

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7,
        )

        model = af.Collection(galaxies=af.Collection(galaxy=af.Model(ag.Galaxy, redshift=0.5)))

        analysis.modify_before_fit(model=model)

        assert (analysis.dataset.noise_map == masked_imaging_7x7.noise_map).all()
        assert (analysis.dataset_modified.noise_map == masked_imaging_7x7.noise_map).all()

        # Hyper galaxy is in model, so identical dataset is used.

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7,
        )

        model = af.Collection(galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, hyper_galaxy=ag.HyperGalaxy)
        ))

        analysis.modify_before_fit(model=model)

        assert (analysis.dataset.noise_map == masked_imaging_7x7.noise_map).all()
        assert (analysis.dataset_modified.noise_map == masked_imaging_7x7.noise_map).all()

        # Hyper galaxy in model but every parameter is fixed (e.g. its an instance) so dataset is updated with
        # noise map using this instance.

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7,
            settings_inversion=ag.SettingsInversion(use_w_tilde=None)
        )

        hyper_galaxy = af.Model(ag.HyperGalaxy)
        hyper_galaxy.contribution_factor = 0.0
        hyper_galaxy.noise_factor = 1.0
        hyper_galaxy.noise_power = 1.0

        model = af.Collection(galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, hyper_galaxy=hyper_galaxy)
        ))

        analysis.modify_before_fit(model=model)

    def test__modify_settings_inversion__before_fit(self, masked_imaging_7x7):

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7,
            settings_inversion=ag.SettingsInversion(use_w_tilde=None)
        )

        assert analysis.settings_inversion.use_w_tilde == None
        assert analysis.settings_inversion_modified == None

        # No hyper galaxy in model, so setting inversions are copied from the input.

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7,
            settings_inversion=ag.SettingsInversion(use_w_tilde=None)
        )

        model = af.Collection(galaxies=af.Collection(galaxy=af.Model(ag.Galaxy, redshift=0.5)))

        analysis.modify_before_fit(model=model)

        assert analysis.settings_inversion.use_w_tilde == None
        assert analysis.settings_inversion_modified.use_w_tilde == None

        # Hyper galaxy in model but every parameter is fixed (e.g. its an instance) so setting inversions are copied
        # from the input.

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7,
            settings_inversion=ag.SettingsInversion(use_w_tilde=None)
        )

        hyper_galaxy = af.Model(ag.HyperGalaxy)
        hyper_galaxy.contribution_factor = 0.0
        hyper_galaxy.noise_factor = 0.0
        hyper_galaxy.noise_power = 1.0

        model = af.Collection(galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, hyper_galaxy=hyper_galaxy)
        ))

        analysis.modify_before_fit(model=model)

        assert analysis.settings_inversion.use_w_tilde == None
        assert analysis.settings_inversion_modified.use_w_tilde == None

        # Hyper galaxy is in model, so setting inversions are copied from the input but w_tilde is set to False.

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7,
            settings_inversion=ag.SettingsInversion(use_w_tilde=None)
        )

        model = af.Collection(galaxies=af.Collection(
            galaxy=af.Model(ag.Galaxy, redshift=0.5, hyper_galaxy=ag.HyperGalaxy)
        ))

        analysis.modify_before_fit(model=model)

        assert analysis.settings_inversion.use_w_tilde == False
        assert analysis.settings_inversion_modified.use_w_tilde == False

    def test__associate_hyper_images(self, masked_imaging_7x7):

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

        result = mock.MockResult(
            instance=instance,
            hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict,
            hyper_model_image=ag.Array2D.full(
                fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0
            ),
        )

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7, hyper_dataset_result=result
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
