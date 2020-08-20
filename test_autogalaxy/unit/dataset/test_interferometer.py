import autogalaxy as ag
import numpy as np
import pytest


class TestMaskedInterferometer:
    def test__masked_dataset_via_autoarray(
        self,
        interferometer_7,
        sub_mask_7x7,
        visibilities_mask_7x2,
        visibilities_7x2,
        noise_map_7x2,
    ):

        masked_interferometer_7 = ag.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=sub_mask_7x7,
            settings=ag.SettingsMaskedInterferometer(
                transformer_class=ag.TransformerDFT
            ),
        )

        assert (
            masked_interferometer_7.visibilities == interferometer_7.visibilities
        ).all()
        assert (masked_interferometer_7.visibilities == visibilities_7x2).all()

        assert (masked_interferometer_7.noise_map == noise_map_7x2).all()

        assert (
            masked_interferometer_7.visibilities_mask
            == np.full(fill_value=False, shape=(7, 2))
        ).all()

        assert (
            masked_interferometer_7.interferometer.uv_wavelengths
            == interferometer_7.uv_wavelengths
        ).all()
        assert (
            masked_interferometer_7.interferometer.uv_wavelengths[0, 0]
            == -55636.4609375
        )

        assert type(masked_interferometer_7.transformer) == ag.TransformerDFT

    def test__inheritance_via_autoarray(
        self,
        interferometer_7,
        sub_mask_7x7,
        visibilities_mask_7x2,
        grid_7x7,
        sub_grid_7x7,
    ):

        masked_interferometer_7 = ag.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7x2,
            real_space_mask=sub_mask_7x7,
            settings=ag.SettingsMaskedInterferometer(grid_class=ag.Grid),
        )

        assert (masked_interferometer_7.grid.in_1d_binned == grid_7x7).all()
        assert (masked_interferometer_7.grid == sub_grid_7x7).all()

        grid = ag.Grid.from_mask(mask=sub_mask_7x7)

        assert (masked_interferometer_7.grid == grid).all()

    def test__different_interferometer_without_mock_objects__customize_constructor_inputs(
        self
    ):
        interferometer = ag.Interferometer(
            visibilities=ag.Visibilities.ones(shape_1d=(19,)),
            noise_map=ag.Visibilities.full(fill_value=2.0, shape_1d=(19,)),
            uv_wavelengths=3.0 * np.ones((19, 2)),
        )

        visibilities_mask = np.full(fill_value=False, shape=(19,))

        real_space_mask = ag.Mask.unmasked(
            shape_2d=(19, 19), pixel_scales=1.0, invert=True, sub_size=8
        )
        real_space_mask[9, 9] = False

        masked_interferometer = ag.MaskedInterferometer(
            interferometer=interferometer,
            visibilities_mask=visibilities_mask,
            real_space_mask=real_space_mask,
        )

        assert (masked_interferometer.visibilities.in_1d == np.ones((19, 2))).all()
        assert (masked_interferometer.noise_map.in_1d == 2.0 * np.ones((19, 2))).all()
        assert (
            masked_interferometer.interferometer.uv_wavelengths
            == 3.0 * np.ones((19, 2))
        ).all()


class TestSimulatorInterferometer:
    def test__from_plane__same_as_plane_input(self):

        grid = ag.Grid.uniform(shape_2d=(20, 20), pixel_scales=0.05, sub_size=1)

        galaxy_0 = ag.Galaxy(
            redshift=0.5,
            light=ag.lp.EllipticalSersic(intensity=1.0),
            mass=ag.mp.EllipticalIsothermal(einstein_radius=1.6),
        )

        galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.EllipticalSersic(intensity=0.3))

        plane = ag.Plane(redshift=0.75, galaxies=[galaxy_0, galaxy_1])

        simulator = ag.SimulatorInterferometer(
            uv_wavelengths=np.ones(shape=(7, 2)),
            exposure_time_map=ag.Array.full(fill_value=10000.0, shape_2d=grid.shape_2d),
            background_sky_map=ag.Array.full(fill_value=100.0, shape_2d=grid.shape_2d),
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.from_plane_and_grid(plane=plane, grid=grid)

        interferometer_via_image = simulator.from_image(
            image=plane.image_from_grid(grid=grid)
        )

        assert (
            interferometer.visibilities == interferometer_via_image.visibilities
        ).all()
        assert (
            interferometer.uv_wavelengths == interferometer_via_image.uv_wavelengths
        ).all()
        assert (interferometer.noise_map == interferometer_via_image.noise_map).all()

    def test__simulate_interferometer_from_galaxy__source_galaxy__compare_to_interferometer(
        self
    ):

        galaxy_0 = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllipticalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
            ),
        )

        galaxy_1 = ag.Galaxy(
            redshift=0.5,
            light=ag.lp.EllipticalSersic(
                centre=(0.1, 0.1),
                elliptical_comps=(0.096225, -0.055555),
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        grid = ag.Grid.uniform(shape_2d=(11, 11), pixel_scales=0.05, sub_size=1)

        simulator = ag.SimulatorInterferometer(
            uv_wavelengths=np.ones(shape=(7, 2)),
            exposure_time_map=ag.Array.full(fill_value=10000.0, shape_2d=grid.shape_2d),
            background_sky_map=ag.Array.full(fill_value=100.0, shape_2d=grid.shape_2d),
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.from_galaxies_and_grid(
            galaxies=[galaxy_0, galaxy_1], grid=grid
        )

        plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

        interferometer_via_image = simulator.from_image(
            image=plane.image_from_grid(grid=grid)
        )

        assert interferometer.visibilities == pytest.approx(
            interferometer_via_image.visibilities, 1.0e-4
        )
        assert (
            interferometer.uv_wavelengths == interferometer_via_image.uv_wavelengths
        ).all()
        assert (interferometer_via_image.noise_map == interferometer.noise_map).all()
