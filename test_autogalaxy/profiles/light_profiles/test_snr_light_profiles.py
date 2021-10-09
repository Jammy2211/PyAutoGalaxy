import numpy as np

import autogalaxy as ag

from autogalaxy.profiles.light_profiles.snr_light_profiles import SNRCalc

from autogalaxy.mock.mock import MockLightProfile


def test__image_used_to_set_intensity():

    image_2d = np.array([[1.0, 2.0], [3.0, 4.0]])

    light_profile = MockLightProfile(image_2d=image_2d)

    snr_calc = SNRCalc(light_profile=light_profile, signal_to_noise_ratio=1.0)

    snr_calc.set_intensity_from(grid=None, exposure_time=10.0)

    print(light_profile.intensity)


def test__signal_to_noise_via_simulator_correct():

    background_sky_level = 10.0
    exposure_time = 300.0

    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    sersic = ag.lp_snr.EllSersic(signal_to_noise_ratio=10.0)

    sersic.snr_calc.set_intensity_from(
        grid=grid,
        exposure_time=exposure_time,
        background_sky_level=background_sky_level,
    )

    simulator = ag.SimulatorImaging(
        exposure_time=exposure_time,
        noise_seed=1,
        background_sky_level=background_sky_level,
    )

    imaging = simulator.via_galaxies_from(
        grid=grid, galaxies=[ag.Galaxy(redshift=0.5, light=sersic)]
    )

    assert 9.0 < imaging.signal_to_noise_max < 11.0
