from __future__ import annotations
from typing import List, Optional

from autoconf.fitsable import ndarray_via_hdu_from

import autofit as af
import autoarray as aa

from autoarray.mask.mask_2d import Mask2DKeys

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages


def mask_header_from(fit):
    """
    Returns the mask, header and pixel scales of the `PyAutoFit` `Fit` object.

    These quantities are commonly loaded during the aggregator interface therefore this method is used to
    avoid code duplication.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database.


    Returns
    -------
    The mask, header and pixel scales of the `PyAutoFit` `Fit` object.
    """

    header = aa.Header(header_sci_obj=fit.value(name="dataset")[0].header)
    pixel_scales = (
        header.header_sci_obj[Mask2DKeys.PIXSCAY.value],
        header.header_sci_obj[Mask2DKeys.PIXSCAY.value],
    )
    origin = (
        header.header_sci_obj[Mask2DKeys.ORIGINY.value],
        header.header_sci_obj[Mask2DKeys.ORIGINX.value],
    )
    mask = aa.Mask2D(
        mask=ndarray_via_hdu_from(fit.value(name="dataset")[0]),
        pixel_scales=pixel_scales,
        origin=origin,
    )

    return mask, header


def adapt_images_from(
    fit: af.Fit,
) -> List[AdaptImages]:
    """
    Updates adaptive images when loading the galaxies from a `PyAutoFit` loaded directory `Fit` or sqlite
    database `Fit` object.

    This function ensures that if adaptive features (e.g. an `Hilbert` image-mesh) are used in a model-fit,
    they work when using the database to load and inspect the results of the model-fit.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.

    Returns
    -------
    A list of galaxies associated with a sample of the non-linear search with adaptive images associated with them.
    """

    fit_list = [fit] if not fit.children else fit.children

    if fit.value(name="adapt_images") is None:
        return [None] * len(fit_list)

    adapt_images_list = []

    for fit in fit_list:
        mask, header = mask_header_from(fit=fit)

        galaxy_name_image_dict = {}

        for i, value in enumerate(fit.value(name="adapt_images")[1:]):
            adapt_image = aa.Array2D.no_mask(
                values=ndarray_via_hdu_from(value),
                pixel_scales=mask.pixel_scales,
                header=header,
                origin=mask.origin,
            )
            adapt_image = adapt_image.apply_mask(mask=mask)

            galaxy_name_image_dict[value.header["EXTNAME"].lower()] = adapt_image

        instance = fit.model.instance_from_prior_medians(ignore_prior_limits=True)

        adapt_images = AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

        adapt_images = adapt_images.updated_via_instance_from(
            instance=instance,
            mask=mask,
        )

        adapt_images_list.append(adapt_images)

    return adapt_images_list


def mesh_grids_of_planes_list_from(
    fit: af.Fit, total_fits: int, use_preloaded_grid: bool
) -> List[Optional[aa.Grid2D]]:
    """
    Returns the image-plane pixelization grid(s) used by the fit.

    A subset of image-mesh objects (e.g. `Hilbert`, `KMeans`) create the grid of points that act as the mesh
    centres (e.g. the centers of Voronoi cells) in the image-plane. For lensing calculations this may then be
    traced to the source-plane to form the pixelization.

    This calculation can depend on the library used to determine the image-plane grid and may have a random element
    associated with it. This means that performing an analysis on a super computer and then downloading the results
    for inspection on your laptop may produce different image-plane grids, changing the result and quantities like
    the `log_likelihood`.

    By storing this grid as a result in the `files` folder and loading it via the database before creating a fit
    this possible mismatch is removed, ensuring results on a local computer are identical to those computer elsewhere.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry which has been loaded from
        an output directory or from an sqlite database..
    total_fits
        The total number of `Analysis` objects summed to create the fit.
    use_preloaded_grid
        Certain pixelization's construct their mesh in the source-plane from a stochastic KMeans algorithm. This grid
        may be output to hard-disk after the model-fit and loaded via the database to ensure the same grid is used
        as the fit.

    Returns
    -------
    The list of image-plane mesh centres used when creating the overall fit.
    """

    if use_preloaded_grid:
        if not fit.children:
            return [fit.value(name="preload_mesh_grids_of_planes")]
        else:
            try:
                return fit.child_values(name="preload_mesh_grids_of_planes")
            except AttributeError:
                return [None] * total_fits
    else:
        return [None] * total_fits


def preloads_from(
    preloads_cls,
    use_preloaded_grid: bool,
    mesh_grids_of_planes: List,
    use_w_tilde: Optional[bool] = False,
) -> aa.Preloads:
    """
    Returns a `Preloads` object associated with a fit loaded via the database.

    When loading results via the database, the preloads class may have certain attributes associated with it
    in order to perform the fit. The main purpose is to use the same image-plane mesh centres for pixelization where
    the mesh is computed in the image-plane (see `agg_util.mesh_grids_of_planes_list_from`).

    The preloads may also switch off `w_tilde` so fits are computed faster locally as they do not need to recompute
    w_tilde.

    Parameters
    ----------
    preloads_cls
        The `Preloads` object used to create the preloads (this varies across
        projects like `autogalaxy` and `autolens`).
    use_preloaded_grid
        Certain pixelization's construct their mesh in the source-plane from a stochastic KMeans algorithm. This grid
        may be output to hard-disk after the model-fit and loaded via the database to ensure the same grid is used
        as the fit.
    mesh_grids_of_planes
        The list of image-plane mesh centres used when creating the overall fit which are associated with the
        preloads.
    use_w_tilde
        Whether to use the w-tilde formalism when recomputing fits.

    Returns
    -------
    The preloads object associated with the fit loaded via the database.
    """
    preloads = preloads_cls()

    if use_preloaded_grid:
        if mesh_grids_of_planes is not None:
            preloads = preloads_cls(
                image_plane_mesh_grid_pg_list=mesh_grids_of_planes,
                use_w_tilde=use_w_tilde,
            )

            if len(preloads.image_plane_mesh_grid_pg_list) == 2:
                if type(preloads.image_plane_mesh_grid_pg_list[1]) != list:
                    preloads.image_plane_mesh_grid_pg_list[1] = [
                        preloads.image_plane_mesh_grid_pg_list[1]
                    ]

    return preloads
