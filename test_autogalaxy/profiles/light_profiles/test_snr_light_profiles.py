import autogalaxy as ag


def test__signal_to_noise_via_simulator_correct():

    background_sky_level = 10.0
    exposure_time = 300.0

    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    sersic = ag.lp_snr.EllSersic(signal_to_noise_ratio=10.0)

    sersic.set_intensity_from(
        grid=grid,
        exposure_time=exposure_time,
        background_sky_level=background_sky_level,
    )

    psf = ag.Kernel2D.manual_native(array=[[1.0]], pixel_scales=0.2)

    simulator = ag.SimulatorImaging(
        psf=psf,
        exposure_time=exposure_time,
        noise_seed=1,
        background_sky_level=background_sky_level,
    )

    imaging = simulator.via_galaxies_from(
        grid=grid, galaxies=[ag.Galaxy(redshift=0.5, light=sersic)]
    )

    assert 9.0 < imaging.signal_to_noise_max < 11.0
