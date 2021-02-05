import os

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt


def pixel_scale_from_instrument(instrument):
    """
    Returns the pixel scale from an instrument type based on real observations.

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
    """
    Returns the `Grid2D` from an instrument type based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """
    if instrument in "vro":
        return ag.Grid2DIterate.uniform(shape_native=(80, 80), pixel_scales=0.2)
    elif instrument in "euclid":
        return ag.Grid2DIterate.uniform(shape_native=(120, 120), pixel_scales=0.1)
    elif instrument in "hst":
        return ag.Grid2DIterate.uniform(shape_native=(200, 200), pixel_scales=0.05)
    elif instrument in "hst_up":
        return ag.Grid2DIterate.uniform(shape_native=(300, 300), pixel_scales=0.03)
    elif instrument in "ao":
        return ag.Grid2DIterate.uniform(shape_native=(800, 800), pixel_scales=0.01)
    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def psf_from_instrument(instrument):
    """
    Returns the *PSF* from an instrument type based on real observations.

    These options are representative of VRO, Euclid, HST, over-sampled HST and Adaptive Optics image.

    Parameters
    ----------
    instrument : str
        A string giving the resolution of the desired instrument (VRO | Euclid | HST | HST_Up | AO).
    """
    if instrument in "vro":
        return ag.Kernel2D.from_gaussian(
            shape_native=(31, 31), sigma=0.5, pixel_scales=0.2, renormalize=True
        )

    elif instrument in "euclid":
        return ag.Kernel2D.from_gaussian(
            shape_native=(31, 31), sigma=0.1, pixel_scales=0.1, renormalize=True
        )
    elif instrument in "hst":
        return ag.Kernel2D.from_gaussian(
            shape_native=(31, 31), sigma=0.05, pixel_scales=0.05, renormalize=True
        )
    elif instrument in "hst_up":
        return ag.Kernel2D.from_gaussian(
            shape_native=(31, 31), sigma=0.05, pixel_scales=0.03, renormalize=True
        )
    elif instrument in "ao":
        return ag.Kernel2D.from_gaussian(
            shape_native=(31, 31), sigma=0.025, pixel_scales=0.01, renormalize=True
        )

    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def simulator_from_instrument(instrument):
    """
    Returns the *Simulator* from an instrument type based on real observations.

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
            exposure_time_map=ag.Array2D.full(
                fill_value=100.0, shape_native=grid.shape_native
            ),
            psf=psf,
            background_sky_map=ag.Array2D.full(
                fill_value=1.0, shape_native=grid.shape_native
            ),
            add_poisson_noise=True,
        )
    elif instrument in "euclid":
        return ag.SimulatorImaging(
            exposure_time_map=ag.Array2D.full(
                fill_value=2260.0, shape_native=grid.shape_native
            ),
            psf=psf,
            background_sky_map=ag.Array2D.full(
                fill_value=1.0, shape_native=grid.shape_native
            ),
            add_poisson_noise=True,
        )
    elif instrument in "hst":
        return ag.SimulatorImaging(
            exposure_time_map=ag.Array2D.full(
                fill_value=2000.0, shape_native=grid.shape_native
            ),
            psf=psf,
            background_sky_map=ag.Array2D.full(
                fill_value=1.0, shape_native=grid.shape_native
            ),
            add_poisson_noise=True,
        )
    elif instrument in "hst_up":
        return ag.SimulatorImaging(
            exposure_time_map=ag.Array2D.full(
                fill_value=2000.0, shape_native=grid.shape_native
            ),
            psf=psf,
            background_sky_map=ag.Array2D.full(
                fill_value=1.0, shape_native=grid.shape_native
            ),
            add_poisson_noise=True,
        )
    elif instrument in "ao":
        return ag.SimulatorImaging(
            exposure_time_map=ag.Array2D.full(
                fill_value=1000.0, shape_native=grid.shape_native
            ),
            psf=psf,
            background_sky_map=ag.Array2D.full(
                fill_value=1.0, shape_native=grid.shape_native
            ),
            add_poisson_noise=True,
        )
    else:
        raise ValueError("An invalid instrument was entered - ", instrument)


def simulate_imaging_from_instrument(data_name, instrument, galaxies):

    # Simulate the imaging data, remembering that we use a special image which ensures edge-effects don't
    # degrade our modeling of the telescope optics (e.ag. the PSF convolution).

    grid = grid_from_instrument(instrument=instrument)

    simulator = simulator_from_instrument(instrument=instrument)

    # Use the input galaxies to setup a plane, which will generate the image for the simulated imaging data.
    plane = ag.Plane(galaxies=galaxies)

    imaging = simulator.from_plane_and_grid(plane=plane, grid=grid)

    # Now, lets output this simulated imaging-data to the test_autoarray/simulator folder.
    test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

    dataset_path = f"dataset/imaging/{data_name}/{instrument}"

    imaging.output_to_fits(
        image_path=path.join(dataset_path, "image.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        overwrite=True,
    )

    plotter = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))
    plotter = aplt.MatPlot2D(output=aplt.Output(path=dataset_path, format="png"))

    aplt.Imaging.subplot_imaging(imaging=imaging, plotter=plotter)

    aplt.imaging.individual(
        imaging=imaging,
        image=True,
        noise_map=True,
        psf=True,
        signal_to_noise_map=True,
        plotter=plotter,
    )

    aplt.plane.image(plane=plane, grid=grid, plotter=plotter)


def load_test_imaging(instrument, data_name, name=None):

    test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))

    pixel_scales = pixel_scale_from_instrument(instrument=instrument)

    dataset_path = f"dataset/imaging/{data_name}/{instrument}"

    return ag.Imaging.from_fits(
        image_path=path.join(dataset_path, "image.fits"),
        psf_path=path.join(dataset_path, "psf.fits"),
        noise_map_path=path.join(dataset_path, "noise_map.fits"),
        pixel_scales=pixel_scales,
        name=name,
    )
