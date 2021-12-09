import autogalaxy as ag
import numpy as np
import pytest


class TestSimulatorInterferometer:
    def test__from_plane__same_as_plane_input(self):

        grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05, sub_size=1)

        galaxy_0 = ag.Galaxy(
            redshift=0.5,
            light=ag.lp.EllSersic(intensity=1.0),
            mass=ag.mp.EllIsothermal(einstein_radius=1.6),
        )

        galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.EllSersic(intensity=0.3))

        plane = ag.Plane(redshift=0.75, galaxies=[galaxy_0, galaxy_1])

        simulator = ag.SimulatorInterferometer(
            uv_wavelengths=np.ones(shape=(7, 2)),
            exposure_time=10000.0,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.via_plane_from(plane=plane, grid=grid)

        interferometer_via_image = simulator.via_image_from(
            image=plane.image_2d_from(grid=grid)
        )

        assert (
            interferometer.visibilities == interferometer_via_image.visibilities
        ).all()
        assert (
            interferometer.uv_wavelengths == interferometer_via_image.uv_wavelengths
        ).all()
        assert (interferometer.noise_map == interferometer_via_image.noise_map).all()

    def test__simulate_interferometer_from_galaxy__source_galaxy__compare_to_interferometer(
        self,
    ):

        galaxy_0 = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.6, elliptical_comps=(0.17647, 0.0)
            ),
        )

        galaxy_1 = ag.Galaxy(
            redshift=0.5,
            light=ag.lp.EllSersic(
                centre=(0.1, 0.1),
                elliptical_comps=(0.096225, -0.055555),
                intensity=0.3,
                effective_radius=1.0,
                sersic_index=2.5,
            ),
        )

        grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.05, sub_size=1)

        simulator = ag.SimulatorInterferometer(
            uv_wavelengths=np.ones(shape=(7, 2)),
            exposure_time=10000.0,
            noise_sigma=0.1,
            noise_seed=1,
        )

        interferometer = simulator.via_galaxies_from(
            galaxies=[galaxy_0, galaxy_1], grid=grid
        )

        plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

        interferometer_via_image = simulator.via_image_from(
            image=plane.image_2d_from(grid=grid)
        )

        assert interferometer.visibilities == pytest.approx(
            interferometer_via_image.visibilities, 1.0e-4
        )
        assert (
            interferometer.uv_wavelengths == interferometer_via_image.uv_wavelengths
        ).all()
        assert (interferometer_via_image.noise_map == interferometer.noise_map).all()
