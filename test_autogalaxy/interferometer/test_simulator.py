import autogalaxy as ag
import numpy as np
import pytest


def test__from_plane__same_as_plane_input():
    grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05)

    galaxy_0 = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(intensity=1.0),
        mass=ag.mp.Isothermal(einstein_radius=1.6),
    )

    galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.Sersic(intensity=0.3))

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

    simulator = ag.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        exposure_time=10000.0,
        noise_sigma=0.1,
        noise_seed=1,
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    interferometer_via_image = simulator.via_image_from(
        image=galaxies.image_2d_from(grid=grid)
    )

    assert (dataset.data == interferometer_via_image.visibilities).all()
    assert (dataset.uv_wavelengths == interferometer_via_image.uv_wavelengths).all()
    assert (dataset.noise_map == interferometer_via_image.noise_map).all()


def test__simulate_interferometer_from_galaxy__source_galaxy__compare_to_interferometer():
    galaxy_0 = ag.Galaxy(
        redshift=0.5,
        mass=ag.mp.Isothermal(
            centre=(0.0, 0.0), einstein_radius=1.6, ell_comps=(0.17647, 0.0)
        ),
    )

    galaxy_1 = ag.Galaxy(
        redshift=0.5,
        light=ag.lp.Sersic(
            centre=(0.1, 0.1),
            ell_comps=(0.096225, -0.055555),
            intensity=0.3,
            effective_radius=1.0,
            sersic_index=2.5,
        ),
    )

    grid = ag.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.05)

    simulator = ag.SimulatorInterferometer(
        uv_wavelengths=np.ones(shape=(7, 2)),
        exposure_time=10000.0,
        noise_sigma=0.1,
        noise_seed=1,
    )

    dataset = simulator.via_galaxies_from(galaxies=[galaxy_0, galaxy_1], grid=grid)

    galaxies = ag.Galaxies(galaxies=[galaxy_0, galaxy_1])

    interferometer_via_image = simulator.via_image_from(
        image=galaxies.image_2d_from(grid=grid)
    )

    assert dataset.data == pytest.approx(interferometer_via_image.visibilities, 1.0e-4)
    assert (dataset.uv_wavelengths == interferometer_via_image.uv_wavelengths).all()
    assert (interferometer_via_image.noise_map == dataset.noise_map).all()
