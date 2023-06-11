import numpy as np
from typing import List

from autoconf import conf

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plane.plane import Plane

from autogalaxy import exc


def plane_image_from(
    galaxies: List[Galaxy],
    grid: aa.Grid2D,
    buffer: float = 1.0e-2,
    zoom_to_brightest: bool = True,
) -> aa.Array2D:
    """
    Returns the plane image of a list of galaxies, by summing their individual images.

    For lensing calculations performed by **PyAutoLens**, this function is used to return the unleensed image
    source-plane galaxies.

    By default, an adaptive grid is used to determine the grid that the images of the galaxies are computed on.
    This grid adapts its dimensions to capture the brightest regions of the image, ensuring that visualization of
    the plane-image is focused entirely on where the galaxies are brightest.

    This adaptive grid is based on determining the size of the grid that contains all pixels with an
    input % (typically 99%) of the total flux of the brightest pixel in the image.

    The adaptive grid can be disabled such that the input grid is used to compute the image of the galaxies.

    Parameters
    ----------
    galaxies
        The list of galaxies whose images are summed to compute the plane image.
    grid
        The grid of (y,x) coordinates for which the image of the galaxies is computed on, or from which the adaptive
        grid is derived.
    buffer
        The buffer around the adaptive grid that is used to ensure the image of the galaxies is not cut off.
    zoom_to_brightest
        If True, an adaptive grid is used to compute the image of the galaxies which zooms in on the brightest
        regions of the image. If False, the input grid is used.

    Returns
    -------
    The plane image of the galaxies, which is the sum of their individual images.
    """

    shape = grid.shape_native

    if zoom_to_brightest:
        try:
            image = sum(map(lambda g: g.image_2d_from(grid=grid), galaxies))
            image = image.native

            zoom_percent = conf.instance["visualize"]["general"]["zoom"][
                "plane_percent"
            ]

            fractional_value = np.max(image) * zoom_percent

            fractional_bool = image > fractional_value

            true_indices = np.argwhere(fractional_bool)

            y_max_pix = np.min(true_indices[:, 0])
            y_min_pix = np.max(true_indices[:, 0])
            x_min_pix = np.min(true_indices[:, 1])
            x_max_pix = np.max(true_indices[:, 1])

            grid = grid.native

            extent = (
                grid[0, x_min_pix][1] - buffer,
                grid[0, x_max_pix][1] + buffer,
                grid[y_min_pix, 0][0] - buffer,
                grid[y_max_pix, 0][0] + buffer,
            )

            extent = aa.util.geometry.extent_symmetric_from(extent=extent)

            pixel_scales = (
                float((extent[3] - extent[2]) / shape[0]),
                float((extent[1] - extent[0]) / shape[1]),
            )
            origin = ((extent[3] + extent[2]) / 2.0, (extent[1] + extent[0]) / 2.0)

            grid = aa.Grid2D.uniform(
                shape_native=grid.shape_native,
                pixel_scales=pixel_scales,
                sub_size=1,
                origin=origin,
            )
        except ValueError:
            pass

    image = sum(map(lambda g: g.image_2d_from(grid=grid), galaxies))

    return aa.Array2D.no_mask(
        values=image.native, pixel_scales=grid.pixel_scales, origin=grid.origin
    )


def ordered_plane_redshifts_from(galaxies):
    """
    Given a list of galaxies (with redshifts), return a list of the redshifts in ascending order.

    If two or more galaxies have the same redshift that redshift is not double counted.

    Parameters
    ----------
    galaxies : [Galaxy]
        The list of galaxies in the ray-tracing calculation.
    """
    ordered_galaxies = sorted(
        galaxies, key=lambda galaxy: galaxy.redshift, reverse=False
    )

    # Ideally we'd extract the planes_red_Shfit order from the list above. However, I dont know how to extract it
    # Using a list of class attributes so make a list of redshifts for now.

    galaxy_redshifts = list(map(lambda galaxy: galaxy.redshift, ordered_galaxies))
    return [
        redshift
        for i, redshift in enumerate(galaxy_redshifts)
        if redshift not in galaxy_redshifts[:i]
    ]


def ordered_plane_redshifts_with_slicing_from(
    lens_redshifts, planes_between_lenses, source_plane_redshift
):
    """Given a set of lens plane redshifts, the source-plane redshift and the number of planes between each, setup the \
    plane redshifts using these values. A lens redshift corresponds to the 'main' lens galaxy(s),
    whereas the slices collect line-of-sight halos over a range of redshifts.

    The source-plane redshift is removed from the ordered plane redshifts that are returned, so that galaxies are not \
    planed at the source-plane redshift.

    For example, if the main plane redshifts are [1.0, 2.0], and the bin sizes are [1,3], the following redshift \
    slices for planes will be used:

    z=0.5
    z=1.0
    z=1.25
    z=1.5
    z=1.75
    z=2.0

    Parameters
    ----------
    lens_redshifts : [float]
        The redshifts of the main-planes (e.g. the lens galaxy), which determine where redshift intervals are placed.
    planes_between_lenses : [int]
        The number of slices between each main plane. The first entry in this list determines the number of slices \
        between Earth (redshift 0.0) and main plane 0, the next between main planes 0 and 1, etc.
    source_plane_redshift
        The redshift of the source-plane, which is input explicitly to ensure galaxies are not placed in the \
        source-plane.
    """

    # Check that the number of slices between lens planes is equal to the number of intervals between the lens planes.
    if len(lens_redshifts) != len(planes_between_lenses) - 1:
        raise exc.PlaneException(
            "The number of lens_plane_redshifts input is not equal to the number of "
            "slices_between_lens_planes+1."
        )

    plane_redshifts = []

    # Add redshift 0.0 and the source plane redshifit to the lens plane redshifts, so that calculation below can use
    # them when dividing slices. These will be removed by the return function at the end from the plane redshifts.

    lens_redshifts.insert(0, 0.0)
    lens_redshifts.append(source_plane_redshift)

    for lens_plane_index in range(1, len(lens_redshifts)):
        previous_plane_redshift = lens_redshifts[lens_plane_index - 1]
        plane_redshift = lens_redshifts[lens_plane_index]
        slice_total = planes_between_lenses[lens_plane_index - 1]
        plane_redshifts += list(
            np.linspace(previous_plane_redshift, plane_redshift, slice_total + 2)
        )[1:]

    return plane_redshifts[0:-1]


def galaxies_in_redshift_ordered_planes_from(galaxies, plane_redshifts):
    """Given a list of galaxies (with redshifts), return a list of the galaxies where each entry contains a list \
    of galaxies at the same redshift in ascending redshift order.

    Parameters
    ----------
    galaxies : [Galaxy]
        The list of galaxies in the ray-tracing calculation.
    """

    galaxies_in_redshift_ordered_planes = [[] for i in range(len(plane_redshifts))]

    for galaxy in galaxies:
        index = (np.abs(np.asarray(plane_redshifts) - galaxy.redshift)).argmin()

        galaxies_in_redshift_ordered_planes[index].append(galaxy)

    return galaxies_in_redshift_ordered_planes


def planes_via_galaxies_from(galaxies, run_time_dict=None, plane_cls=Plane):
    plane_redshifts = ordered_plane_redshifts_from(galaxies=galaxies)

    galaxies_in_planes = galaxies_in_redshift_ordered_planes_from(
        galaxies=galaxies, plane_redshifts=plane_redshifts
    )

    planes = []

    for plane_index in range(0, len(plane_redshifts)):
        planes.append(
            plane_cls(
                galaxies=galaxies_in_planes[plane_index], run_time_dict=run_time_dict
            )
        )

    return planes
