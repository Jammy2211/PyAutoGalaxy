from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy

import autofit as af
import autoarray as aa


def galaxies_with_adapt_images_from(
    fit: af.Fit, galaxies: List[Galaxy]
) -> List[Galaxy]:
    """
    Associates adaptive images with galaxies when loading the galaxies from a `PyAutoFit` sqlite database `Fit` object.

    The adapt galaxies are stored in the fit's `files/adapt` folder, which includes:

    - The `adapt_model_image` associated with every galaxy (`adapt/adapt_model_image.fits`).
    - The `adapt_galaxy_image` associated with every galaxy (e.g. `adapt/("galaxies", "g0").fits`).
    - The `adapt_galaxy_keys` that are used to associate the `adapt_galaxy_image` with each galaxy.

    This function ensures that if adaptive features (e.g. a `VoronoiBrightnessImage` mesh) are used in a model-fit,
    they work when using the database to load and inspect the results of the model-fit.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.

    Returns
    -------
    A list of galaxies associated with a sample of the non-linear search with adaptive images associated with them.
    """

    from autogalaxy.galaxy.galaxy import Galaxy

    adapt_model_image = fit.value(name="adapt.adapt_model_image")

    if adapt_model_image is None:
        return galaxies

    adapt_model_image = aa.Array2D.from_primary_hdu(primary_hdu=adapt_model_image)

    mask = aa.Mask2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.mask"))

    adapt_model_image = adapt_model_image.apply_mask(mask=mask)

    adapt_galaxy_keys = fit.value(name="adapt.adapt_galaxy_keys")

    adapt_galaxy_image_path_dict = {}

    for key in adapt_galaxy_keys:
        adapt_galaxy_image_path_dict[key] = aa.Array2D.from_primary_hdu(
            primary_hdu=fit.value(name=f"adapt.{key}")
        )
        adapt_galaxy_image_path_dict[key] = adapt_galaxy_image_path_dict[
            key
        ].apply_mask(mask=mask)

    model = fit.value(name="model")
    instance = model.instance_from_prior_medians(ignore_prior_limits=True)
    galaxy_path_list = [
        gal[0] for gal in instance.path_instance_tuples_for_class(Galaxy)
    ]

    galaxies_with_adapt = []

    for galaxy_path, galaxy in zip(galaxy_path_list, galaxies):
        if str(galaxy_path) in adapt_galaxy_image_path_dict:
            galaxy.adapt_model_image = adapt_model_image
            galaxy.adapt_galaxy_image = adapt_galaxy_image_path_dict[str(galaxy_path)]

        galaxies_with_adapt.append(galaxy)

    return galaxies_with_adapt


def sparse_grids_of_planes_list_from(
    fit: af.Fit, total_fits: int, use_preloaded_grid: bool
) -> List[Optional[aa.Grid2D]]:
    """
    Returns the image-plane pixelization grid(s) used by the fit.

    A subset of mesh objects (e.g. `VoronoiBrightnessImage`) create the grid of points that act as the mesh
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
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
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
            return [fit.value(name="preload_sparse_grids_of_planes")]
        else:
            try:
                return fit.child_values(name="preload_sparse_grids_of_planes")
            except AttributeError:
                return [None] * total_fits
    else:
        return [None] * total_fits


def preloads_from(
    preloads_cls,
    use_preloaded_grid: bool,
    sparse_grids_of_planes: List,
    use_w_tilde: Optional[bool] = False,
) -> aa.Preloads:
    """
    Returns a `Preloads` object associated with a fit loaded via the database.

    When loading results via the database, the preloads class may have certain attributes associated with it
    in order to perform the fit. The main purpose is to use the same image-plane mesh centres for pixelization where
    the mesh is computed in the image-plane (see `agg_util.sparse_grids_of_planes_list_from`).

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
    sparse_grids_of_planes
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
        if sparse_grids_of_planes is not None:
            preloads = preloads_cls(
                sparse_image_plane_grid_pg_list=sparse_grids_of_planes,
                use_w_tilde=use_w_tilde,
            )

            if len(preloads.sparse_image_plane_grid_pg_list) == 2:
                if type(preloads.sparse_image_plane_grid_pg_list[1]) != list:
                    preloads.sparse_image_plane_grid_pg_list[1] = [
                        preloads.sparse_image_plane_grid_pg_list[1]
                    ]

    return preloads
