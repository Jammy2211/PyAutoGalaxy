import autoarray as aa
import autoastro as aast
import autofit as af
import numpy as np
import pytest

from test_autoastro.mock.mock_galaxy import MockGalaxy


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    af.conf.instance = af.conf.default


class TestLikelihood:
    def test__1x1_image__light_profile_fits_data_perfectly__lh_is_noise(self):
        image = aa.Array.ones(shape_2d=(3, 3), pixel_scales=1.0)

        noise_map = aa.Array.ones(shape_2d=(3, 3), pixel_scales=1.0)

        galaxy_data = aast.GalaxyData(
            image=image, noise_map=noise_map, pixel_scales=3.0
        )

        mask = aa.Mask.manual(
            mask_2d=np.array(
                [[True, True, True], [True, False, True], [True, True, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )
        g0 = MockGalaxy(value=1.0)

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_image=True
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_convergence=True
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_potential=True
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_deflections_y=True
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_deflections_x=True
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.likelihood == -0.5 * np.log(2 * np.pi * 1.0)

    def test__1x2_image__noise_not_1__alls_correct(self):
        image = aa.Array.full(fill_value=5.0, shape_2d=(3, 4), pixel_scales=1.0)
        image[6] = 4.0

        noise_map = aa.Array.full(fill_value=2.0, shape_2d=(3, 4), pixel_scales=1.0)

        galaxy_data = aast.GalaxyData(
            image=image, noise_map=noise_map, pixel_scales=3.0
        )

        mask = aa.Mask.manual(
            mask_2d=np.array(
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

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_image=True
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )

        assert fit.model_galaxies == [g0]
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_convergence=True
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_potential=True
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.model_galaxies == [g0]
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_deflections_y=True
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=galaxy_data, mask=mask, use_deflections_x=True
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[g0]
        )
        assert fit.chi_squared == (25.0 / 4.0)
        assert fit.reduced_chi_squared == (25.0 / 4.0) / 2.0
        assert fit.likelihood == -0.5 * (
            (25.0 / 4.0) + 2.0 * np.log(2 * np.pi * 2.0 ** 2)
        )


class TestCompareToManual:
    def test__image(self, gal_data_7x7, sub_mask_7x7):
        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_image=True
        )

        galaxy = aast.Galaxy(
            redshift=0.5,
            light=aast.lp.SphericalSersic(centre=(1.0, 2.0), intensity=1.0),
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[galaxy]
        )

        assert fit.model_galaxies == [galaxy]

        model_data = galaxy.profile_image_from_grid(grid=masked_galaxy_dataset.grid)

        residual_map = aa.util.fit.residual_map_from_data_and_model_data(
            data=masked_galaxy_dataset.image, model_data=model_data.in_1d_binned
        )

        assert residual_map == pytest.approx(fit.residual_map, 1e-4)

        chi_squared_map = aa.util.fit.chi_squared_map_from_residual_map_and_noise_map(
            residual_map=residual_map, noise_map=masked_galaxy_dataset.noise_map
        )

        assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

        chi_squared = aa.util.fit.chi_squared_from_chi_squared_map(
            chi_squared_map=chi_squared_map
        )

        noise_normalization = aa.util.fit.noise_normalization_from_noise_map(
            noise_map=masked_galaxy_dataset.noise_map
        )

        likelihood = aa.util.fit.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert likelihood == pytest.approx(fit.likelihood, 1e-4)

    def test__convergence(self, gal_data_7x7, sub_mask_7x7):
        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_convergence=True
        )

        galaxy = aast.Galaxy(
            redshift=0.5,
            mass=aast.mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0),
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[galaxy]
        )

        assert fit.model_galaxies == [galaxy]

        model_data = galaxy.convergence_from_grid(grid=masked_galaxy_dataset.grid)

        residual_map = aa.util.fit.residual_map_from_data_and_model_data(
            data=masked_galaxy_dataset.image, model_data=model_data.in_1d_binned
        )
        assert residual_map == pytest.approx(fit.residual_map, 1e-4)

        chi_squared_map = aa.util.fit.chi_squared_map_from_residual_map_and_noise_map(
            residual_map=residual_map, noise_map=masked_galaxy_dataset.noise_map
        )
        assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

        chi_squared = aa.util.fit.chi_squared_from_chi_squared_map(
            chi_squared_map=chi_squared_map
        )

        noise_normalization = aa.util.fit.noise_normalization_from_noise_map(
            noise_map=masked_galaxy_dataset.noise_map
        )

        likelihood = aa.util.fit.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert likelihood == pytest.approx(fit.likelihood, 1e-4)

    def test__potential(self, gal_data_7x7, sub_mask_7x7):
        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_potential=True
        )

        galaxy = aast.Galaxy(
            redshift=0.5,
            mass=aast.mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0),
        )

        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[galaxy]
        )

        assert fit.model_galaxies == [galaxy]

        model_data = galaxy.potential_from_grid(grid=masked_galaxy_dataset.grid)

        residual_map = aa.util.fit.residual_map_from_data_and_model_data(
            data=masked_galaxy_dataset.image, model_data=model_data.in_1d_binned
        )

        assert residual_map == pytest.approx(fit.residual_map, 1e-4)

        chi_squared_map = aa.util.fit.chi_squared_map_from_residual_map_and_noise_map(
            residual_map=residual_map, noise_map=masked_galaxy_dataset.noise_map
        )

        assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

        chi_squared = aa.util.fit.chi_squared_from_chi_squared_map(
            chi_squared_map=chi_squared_map
        )

        noise_normalization = aa.util.fit.noise_normalization_from_noise_map(
            noise_map=masked_galaxy_dataset.noise_map
        )

        likelihood = aa.util.fit.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert likelihood == pytest.approx(fit.likelihood, 1e-4)

    def test__deflections_y(self, gal_data_7x7, sub_mask_7x7):

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_y=True
        )

        galaxy = aast.Galaxy(
            redshift=0.5,
            mass=aast.mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0),
        )

        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[galaxy]
        )

        assert fit.model_galaxies == [galaxy]

        model_data = galaxy.deflections_from_grid(
            grid=masked_galaxy_dataset.grid
        ).in_1d_binned[:, 0]

        residual_map = aa.util.fit.residual_map_from_data_and_model_data(
            data=masked_galaxy_dataset.image, model_data=model_data
        )

        assert residual_map == pytest.approx(fit.residual_map, 1e-4)

        chi_squared_map = aa.util.fit.chi_squared_map_from_residual_map_and_noise_map(
            residual_map=residual_map, noise_map=masked_galaxy_dataset.noise_map
        )

        assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

        chi_squared = aa.util.fit.chi_squared_from_chi_squared_map(
            chi_squared_map=chi_squared_map
        )

        noise_normalization = aa.util.fit.noise_normalization_from_noise_map(
            noise_map=masked_galaxy_dataset.noise_map
        )

        likelihood = aa.util.fit.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert likelihood == pytest.approx(fit.likelihood, 1e-4)

    def test__deflections_x(self, gal_data_7x7, sub_mask_7x7):

        masked_galaxy_dataset = aast.MaskedGalaxyDataset(
            galaxy_data=gal_data_7x7, mask=sub_mask_7x7, use_deflections_x=True
        )

        galaxy = aast.Galaxy(
            redshift=0.5,
            mass=aast.mp.SphericalIsothermal(centre=(1.0, 2.0), einstein_radius=1.0),
        )
        fit = aast.FitGalaxy(
            masked_galaxy_dataset=masked_galaxy_dataset, model_galaxies=[galaxy]
        )

        assert fit.model_galaxies == [galaxy]

        model_data = galaxy.deflections_from_grid(
            grid=masked_galaxy_dataset.grid
        ).in_1d_binned[:, 1]

        residual_map = aa.util.fit.residual_map_from_data_and_model_data(
            data=masked_galaxy_dataset.image, model_data=model_data
        )

        assert residual_map == pytest.approx(fit.residual_map, 1e-4)

        chi_squared_map = aa.util.fit.chi_squared_map_from_residual_map_and_noise_map(
            residual_map=residual_map, noise_map=masked_galaxy_dataset.noise_map
        )

        assert chi_squared_map == pytest.approx(fit.chi_squared_map, 1e-4)

        chi_squared = aa.util.fit.chi_squared_from_chi_squared_map(
            chi_squared_map=chi_squared_map
        )

        noise_normalization = aa.util.fit.noise_normalization_from_noise_map(
            noise_map=masked_galaxy_dataset.noise_map
        )

        likelihood = aa.util.fit.likelihood_from_chi_squared_and_noise_normalization(
            chi_squared=chi_squared, noise_normalization=noise_normalization
        )

        assert likelihood == pytest.approx(fit.likelihood, 1e-4)
