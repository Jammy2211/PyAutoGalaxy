import os

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt


def pixel_scale_from_instrument(instrument):
    """Determine the pixel scale from an instrument type based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """
    if instrument in "vro":
        return (0.2, 0.2)
    elif instrument in "euclid":
        return (0.1, 0.1)
    elif instrument in "hst":
        return (0.05, 0.05)
    elif instrument in "hst_up":
        return (0.03, 0.03)
    elif instrument in "ao":
        return (0.01, 0.01)
    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def grid_from_instrument(instrument):
    """Determine the *Grid* from an instrument type based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """
    if instrument in "vro":
        return ag.Grid.uniform(shape_2d=(50, 50), pixel_scales=0.2)
    elif instrument in "euclid":
        return ag.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.1)
    elif instrument in "hst":
        return ag.Grid.uniform(shape_2d=(160, 160), pixel_scales=0.05)
    elif instrument in "hst_up":
        return ag.Grid.uniform(shape_2d=(100, 100), pixel_scales=0.03)
    elif instrument in "ao":
        return ag.Grid.uniform(shape_2d=(800, 800), pixel_scales=0.01)
    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def psf_from_instrument(instrument):
    """Determine the *PSF* from an instrument type based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """
    if instrument in "vro":
        return ag.Kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.5, pixel_scales=0.2, renormalize=True
        )

    elif instrument in "euclid":
        return ag.Kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.1, pixel_scales=0.1, renormalize=True
        )
    elif instrument in "hst":
        return ag.Kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.05, pixel_scales=0.05, renormalize=True
        )
    elif instrument in "hst_up":
        return ag.Kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.05, pixel_scales=0.03, renormalize=True
        )
    elif instrument in "ao":
        return ag.Kernel.from_gaussian(
            shape_2d=(31, 31), sigma=0.025, pixel_scales=0.01, renormalize=True
        )

    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def simulator_from_instrument(instrument):
    """Determine the *Simulator* from an instrument type based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """

    grid = grid_from_instrument(instrument=instrument)
    psf = psf_from_instrument(instrument=instrument)

    if instrument in "vro":
        return ag.SimulatorImaging(
            exposure_time_map=ag.Array.full(fill_value=100.0, shape_2d=grid.shape_2d),
            psf=psf,
            background_sky_map=ag.Array.full(fill_value=1.0, shape_2d=grid.shape_2d),
            add_noise=True,
        )
    elif instrument in "euclid":
        return ag.SimulatorImaging(
            exposure_time_map=ag.Array.full(fill_value=2260.0, shape_2d=grid.shape_2d),
            psf=psf,
            background_sky_map=ag.Array.full(fill_value=1.0, shape_2d=grid.shape_2d),
            add_noise=True,
        )
    elif instrument in "hst":
        return ag.SimulatorImaging(
            exposure_time_map=ag.Array.full(fill_value=2000.0, shape_2d=grid.shape_2d),
            psf=psf,
            background_sky_map=ag.Array.full(fill_value=1.0, shape_2d=grid.shape_2d),
            add_noise=True,
        )
    elif instrument in "hst_up":
        return ag.SimulatorImaging(
            exposure_time_map=ag.Array.full(fill_value=2000.0, shape_2d=grid.shape_2d),
            psf=psf,
            background_sky_map=ag.Array.full(fill_value=1.0, shape_2d=grid.shape_2d),
            add_noise=True,
        )
    elif instrument in "ao":
        return ag.SimulatorImaging(
            exposure_time_map=ag.Array.full(fill_value=1000.0, shape_2d=grid.shape_2d),
            psf=psf,
            background_sky_map=ag.Array.full(fill_value=1.0, shape_2d=grid.shape_2d),
            add_noise=True,
        )
    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def simulate_imaging_from_instrument(data_label, instrument, galaxies):

    # Simulate the imaging data, remembering that we use a special image which ensures edge-effects don't
    # degrade our modeling of the telescope optics (e.ag. the PSF convolution).

    grid = grid_from_instrument(instrument=instrument)

    simulator = simulator_from_instrument(instrument=instrument)

    # Use the input galaxies to setup a plane, which will generate the image for the simulated imaging data.
    plane = ag.Plane(galaxies=galaxies)

    imaging = simulator.from_plane_and_grid(plane=plane, grid=grid)

    # Now, lets output this simulated imaging-data to the test_autoarray/simulator folder.
    test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

    dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path, folder_names=["dataset", "imaging", data_label, instrument]
    )

    imaging.output_to_fits(
        image_path=f"{dataset_path}/image.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        overwrite=True,
    )

    plotter = aplt.Plotter(output=aplt.Output(path=dataset_path, format="png"))
    sub_plotter = aplt.SubPlotter(output=aplt.Output(path=dataset_path, format="png"))

    aplt.Imaging.subplot_imaging(imaging=imaging, sub_plotter=sub_plotter)

    aplt.Imaging.individual(
        imaging=imaging,
        plot_image=True,
        plot_noise_map=True,
        plot_psf=True,
        plot_signal_to_noise_map=True,
        plotter=plotter,
    )

    aplt.Plane.profile_image(plane=plane, grid=grid, plotter=plotter)


def simulate_ci_data_from_ci_normalization_region_and_cti_model(
    ci_data_type,
    ci_data_model,
    ci_data_resolution,
    clocker,
    pattern,
    parallel_traps=None,
    parallel_ccd_volume=None,
    serial_traps=None,
    serial_ccd_volume=None,
    read_noise=1.0,
    cosmic_ray_map=None,
):

    shape = simulate_util.shape_from_ci_data_resolution(
        ci_data_resolution=ci_data_resolution
    )

    ci_pre_cti = pattern.simulate_ci_pre_cti(shape=shape)

    simulator = ac.ci.SimulatorCIImaging(read_noise=read_noise)

    imaging = simulator.from_image(
        clocker=clocker,
        ci_pre_cti=ci_pre_cti,
        ci_pattern=pattern,
        parallel_traps=parallel_traps,
        parallel_ccd_volume=parallel_ccd_volume,
        serial_traps=serial_traps,
        serial_ccd_volume=serial_ccd_volume,
        cosmic_ray_map=cosmic_ray_map,
    )

    # Now, lets output this simulated ccd-simulator to the test_autocti/simulator folder.
    test_path = "{}/../".format(os.path.dirname(os.path.realpath(__file__)))

    ci_data_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path,
        folder_names=["dataset", ci_data_type, ci_data_model, ci_data_resolution],
    )

    normalization = str(int(pattern.normalization))

    imaging.output_to_fits(
        image_path=ci_data_path + "image_" + normalization + ".fits",
        noise_map_path=ci_data_path + "noise_map_" + normalization + ".fits",
        ci_pre_cti_path=ci_data_path + "ci_pre_cti_" + normalization + ".fits",
        cosmic_ray_map_path=ci_data_path + "cosmic_ray_map_" + normalization + ".fits",
        overwrite=True,
    )


def load_test_imaging(instrument, data_label, name=None):

    test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

    pixel_scales = pixel_scale_from_instrument(instrument=instrument)

    dataset_path = af.path_util.make_and_return_path_from_path_and_folder_names(
        path=test_path, folder_names=["dataset", "imaging", data_label, instrument]
    )

    return ag.Imaging.from_fits(
        image_path=f"{dataset_path}/image.fits",
        psf_path=f"{dataset_path}/psf.fits",
        noise_map_path=f"{dataset_path}/noise_map.fits",
        pixel_scales=pixel_scales,
        name=name,
    )
