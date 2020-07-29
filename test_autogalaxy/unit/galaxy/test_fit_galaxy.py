from autoconf import conf
import autogalaxy as ag
import numpy as np
import pytest
from test_autogalaxy.mock import MockGalaxy


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    conf.instance = conf.default


class TestLikelihood:
    def test__1x1_image__light_profile_fits_data_perfectly__lh_is_noise(self):
        image = ag.Array.ones(shape_2d=(3, 3), pixel_scales=1.0)

        noise_map = ag.Array.ones(shape_2d=(3, 3), pixel_scales=1.0)

        galaxy_data = ag.GalaxyData(image=image, noise_map=noise_map, pixel_scales=3.0)

        mask = ag.Mask.manual(
            mask=np.array(
                [[True, True, True], [True, False, True], [True, True, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )
        g0 = MockGalaxy(value=1.0)

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_image=True
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.log_likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_convergence=True
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.log_likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_potential=True
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.log_likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_deflections_y=True
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.log_likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_deflections_x=True
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.log_likelihood == -0.5 * np.log(2 * np.pi * 1.0)

    def test__1x2_image__noise_not_1__alls_correct(self):
        image = ag.Array.full(fill_value=5.0, shape_2d=(3, 4), pixel_scales=1.0)
        image[6] = 4.0

        noise_map = ag.Array.full(fill_value=2.0, shape_2d=(3, 4), pixel_scales=1.0)

        galaxy_data = ag.GalaxyData(image=image, noise_map=noise_map, pixel_scales=3.0)

        mask = ag.Mask.manual(
            mask=np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        g0 = MockGalaxy(value=1.0, shape=2)

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_image=True
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )

        assert fit.model_galaxies == [g0]
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.log_likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_convergence=True
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.log_likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_potential=True
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.log_likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_deflections_y=True
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.log_likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_deflections_x=True
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.log_likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )


class TestCompareToManual:
    def test__image(self, gal_data_7x7, sub_mask_7x7):
        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_image=True
        )

        galaxy = ag.Galaxy(
            redshift=0.5, light=ag.lp.SphericalSersic(centre=(1.0, 2.0), intensity=1.0)
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[galaxy]
        )

        assert fit.model_galaxies == [galaxy]

        model_data = galaxy.image_from_grid(grid=masked_galaxy_dataset.grid)

        residual_map = ag.util.fit.residual_map_from(
            data=masked_galaxy_dataset.image, model_data=model_data.in_1d_binned
        )

        assert residual_map == pytest.approx(fit.residual_map, 1e-4)

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=masked_galaxy_dataset.noise_map
        )

        assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=masked_galaxy_dataset.noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)

    def test__convergence(self, gal_data_7x7, sub_mask_7x7):
        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_convergence=True
        )

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0),
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[galaxy]
        )

        assert fit.model_galaxies == [galaxy]

        model_data = galaxy.convergence_from_grid(grid=masked_galaxy_dataset.grid)

        residual_map = ag.util.fit.residual_map_from(
            data=masked_galaxy_dataset.image, model_data=model_data.in_1d_binned
        )
        assert residual_map == pytest.approx(fit.residual_map, 1e-4)

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=masked_galaxy_dataset.noise_map
        )
        assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=masked_galaxy_dataset.noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)

    def test__potential(self, gal_data_7x7, sub_mask_7x7):
        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_potential=True
        )

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0),
        )

        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[galaxy]
        )

        assert fit.model_galaxies == [galaxy]

        model_data = galaxy.potential_from_grid(grid=masked_galaxy_dataset.grid)

        residual_map = ag.util.fit.residual_map_from(
            data=masked_galaxy_dataset.image, model_data=model_data.in_1d_binned
        )

        assert residual_map == pytest.approx(fit.residual_map, 1e-4)

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=masked_galaxy_dataset.noise_map
        )

        assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=masked_galaxy_dataset.noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)

    def test__deflections_y(self, gal_data_7x7, sub_mask_7x7):

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_y=True
        )

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0),
        )

        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[galaxy]
        )

        assert fit.model_galaxies == [galaxy]

        model_data = galaxy.deflections_from_grid(
            grid=masked_galaxy_dataset.grid
        ).in_1d_binned[:, 0]

        residual_map = ag.util.fit.residual_map_from(
            data=masked_galaxy_dataset.image, model_data=model_data
        )

        assert residual_map == pytest.approx(fit.residual_map, 1e-4)

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=masked_galaxy_dataset.noise_map
        )

        assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=masked_galaxy_dataset.noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)

    def test__deflections_x(self, gal_data_7x7, sub_mask_7x7):

        masked_galaxy_dataset = ag.MaskedGalaxyDataset(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_x=True
        )

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0),
        )
        fit = ag.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[galaxy]
        )

        assert fit.model_galaxies == [galaxy]

        model_data = galaxy.deflections_from_grid(
            grid=masked_galaxy_dataset.grid
        ).in_1d_binned[:, 1]

        residual_map = ag.util.fit.residual_map_from(
            data=masked_galaxy_dataset.image, model_data=model_data
        )

        assert residual_map == pytest.approx(fit.residual_map, 1e-4)

        chi_squared_map = ag.util.fit.chi_squared_map_from(
            residual_map=residual_map, noise_map=masked_galaxy_dataset.noise_map
        )

        assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

        chi_squared = ag.util.fit.chi_squared_from(chi_squared_map=chi_squared_map)

        noise_normalization = ag.util.fit.noise_normalization_from(
            noise_map=masked_galaxy_dataset.noise_map
        )

        log_likelihood = ag.util.fit.log_likelihood_from(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert log_likelihood == pytest.approx(fit.log_likelihood, 1e-4)
