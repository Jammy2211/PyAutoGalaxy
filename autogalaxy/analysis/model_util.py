from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from autogalaxy.analysis.analysis import AnalysisDataset
    from autogalaxy.analysis.result import ResultDataset

import autofit as af
import autoarray as aa

from autogalaxy import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


def mesh_list_from(model: af.Collection) -> List[aa.AbstractMesh]:
    """
    For a model containing one or more galaxies, inspect its attributes and return the list of `mesh`'s of each
    `pixelization` of all galaxies. If no galaxy has pixelization an empty list is returned.

    This function expects that the input model is a `Collection` where the first model-component has the
    name `galaxies`, and is itself a `Collection` of `Galaxy` instances. This is the
    standard API for creating a model in PyAutoGalaxy.

    The result of `mesh_from` is used by the preloading to determine whether certain parts of a
    calculation can be cached before the non-linear search begins for efficiency.

    Parameters
    ----------
    model
        Contains the `galaxies` in the model that will be fitted via the non-linear search.

    Returns
    -------
    The `mesh` of a galaxy, provided one galaxy has a `mesh`.
    """

    instance = model.instance_from_prior_medians(ignore_prior_limits=True)

    mesh_list = []

    for galaxy in instance.galaxies:
        pixelization_list = galaxy.cls_list_from(cls=aa.Pixelization)

        for pixelization in pixelization_list:
            if pixelization is not None:
                mesh_list.append(pixelization.mesh)

    return mesh_list


def has_pixelization_from(model: af.Collection) -> bool:
    """
    For a model containing one or more galaxies, inspect its attributes and return `True` if a galaxy has a
    `Pixelization` otherwise return `False`.

    This function expects that the input model is a `Collection` where the first model-component has the
    name `galaxies`, and is itself a `Collection` of `Galaxy` instances. This is the
    standard API for creating a model in PyAutoGalaxy.

    The result of `has_pixelization_from` is used by the preloading to determine whether certain parts of a
    calculation can be cached before the non-linear search begins for efficiency.

    Parameters
    ----------
    model : af.Collection
        Contains the `galaxies` in the model that will be fitted via the non-linear search.

    Returns
    -------
    aa.mesh.Mesh or None:
        The `Pixelization` of a galaxy, provided one galaxy has a `Pixelization`.
    """
    mesh_list = mesh_list_from(model=model)

    return len(mesh_list) > 0


def set_upper_limit_of_pixelization_pixels_prior(
    model: af.Collection,
    pixels_in_mask: int,
    lower_limit_no_pixels_below_mask: int = 10,
):
    """
    If the mesh(es) of pixelizations being fitted in the adapt-model fit is a `VoronoiBrightnessImage` pixelization,
    this function sets the upper limit of its `pixels` prior to the number of data points in the mask.

    This ensures the KMeans algorithm does not raise an exception due to having fewer data points than source pixels.

    Parameters
    ----------
    model
        The adapt model used by the adapt-fit, which models adapt-components like a `Pixelization`..
    pixels_in_mask
        The number of pixels in the mask, which are used to set the upper and lower limits of the priors on the
        number of pixels in the pixelization.
    lower_limit_no_pixels_below_mask
        If the prior lower limit on the pixelization's number of pixels is above the number of pixels in the mask,
        the number of pixels in the mask below which the lower limit is set.
    """

    if not hasattr(model, "galaxies"):
        return

    mesh_list = model.galaxies.models_with_type(
        cls=(
            aa.mesh.DelaunayBrightnessImage,
            aa.mesh.VoronoiBrightnessImage,
            aa.mesh.VoronoiNNBrightnessImage,
        )
    )

    if not mesh_list:
        return

    for mesh in mesh_list:
        if hasattr(mesh.pixels, "upper_limit"):
            if pixels_in_mask < mesh.pixels.upper_limit:
                lower_limit = mesh.pixels.lower_limit

                log_str = (
                    "MODIFY BEFORE FIT -  A pixelization mesh's pixel UniformPrior upper limit"
                    "was greater than the number of pixels in the mask. It has been "
                    "reduced to the number of pixels in the mask.\,"
                )

                if lower_limit > pixels_in_mask:
                    lower_limit = pixels_in_mask - lower_limit_no_pixels_below_mask

                    logger.info(
                        log_str
                        + "MODIFY BEFORE FIT - The pixelization's mesh's pixel UniformPrior lower_limit was "
                        "also above the number of pixels in the mask, and has been reduced"
                        "to the number of pixels in the mask minus 10."
                    )
                else:
                    logger.info(log_str)

                mesh.pixels = af.UniformPrior(
                    lower_limit=lower_limit,
                    upper_limit=pixels_in_mask,
                )


def clean_model_of_adapt_images(model):
    for galaxy in model.galaxies:
        del galaxy.adapt_model_image
        del galaxy.adapt_galaxy_image

    if hasattr(model, "clumps"):
        for clump in model.clumps:
            del clump.adapt_model_image
            del clump.adapt_galaxy_image

    return model


def adapt_model_from(
    setup_adapt,
    result: af.Result,
    pixelization_overwrite=None,
    regularization_overwrite=None,
) -> af.Collection:
    """
    Make a adapt model from the `Result` of a model-fit, where the adapt-model is the maximum log likelihood instance
    of the inferred model but turns the following adapt components of the model to free parameters:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.

    The adapt model is typically used in pipelines to refine and improve an `LEq` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    setup_adapt
        The setup of the adapt fit.
    result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the adapt model.

    Returns
    -------
    af.Collection
        The adapt model, which has an instance of the input results maximum log likelihood model with certain adapt
        model components now free parameters.
    """

    if setup_adapt is None:
        return None

    model = result.instance.as_model(
        model_classes=(aa.AbstractMesh, aa.AbstractRegularization),
        excluded_classes=(aa.reg.ConstantZeroth, aa.reg.Zeroth),
    )

    if not has_pixelization_from(model=model):
        return None

    if pixelization_overwrite:
        model.galaxies.source.pixelization = af.Model(pixelization_overwrite)

    if regularization_overwrite:
        model.galaxies.source.regularization = af.Model(regularization_overwrite)

    if setup_adapt.mesh_pixels_fixed is not None:
        if hasattr(model.galaxies.source.pixelization.mesh, "pixels"):
            model.galaxies.source.pixelization.mesh.pixels = (
                setup_adapt.mesh_pixels_fixed
            )

    model = clean_model_of_adapt_images(model=model)

    return model


def adapt_fit(
    setup_adapt,
    result: af.Result,
    analysis,
    search_previous,
):
    """
    Perform a adapt-fit, which extends a model-fit with an additional fit which fixes the non-pixelization components of the
    model (e.g., `LightProfile`'s, `MassProfile`) to the `Result`'s maximum likelihood fit. The adapt-fit then treats
    only the adaptive pixelization's components as free parameters, which are any of the following model components:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.

    The adapt model is typically used in pipelines to refine and improve an `Inversion` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    setup_adapt
        The setup of the adapt fit.
    result
        The result of a previous `Analysis` search whose maximum log likelihood model forms the basis of the adapt model.
    analysis
        An analysis which is used to fit imaging or interferometer data with a model.

    Returns
    -------
    af.Result
        The result of the adapt model-fit, which has a new attribute `result.adapt` that contains updated parameter
        values for the adaptive pixelization's components for passing to later model-fits.
    """

    if analysis.adapt_model_image is None:
        raise exc.AnalysisException(
            "The analysis class does not have a adapt_model_image attribute, which is required for adapt fitting."
        )

    adapt_model_pix = adapt_model_from(
        setup_adapt=setup_adapt,
        result=result,
    )

    search = setup_adapt.search_pix_cls(
        path_prefix=search_previous.path_prefix_no_unique_tag,
        name=f"{search_previous.paths.name}__adapt",
        unique_tag=search_previous.paths.unique_tag,
        number_of_cores=search_previous.number_of_cores,
        **setup_adapt.search_pix_dict,
    )

    set_upper_limit_of_pixelization_pixels_prior(
        model=adapt_model_pix, pixels_in_mask=result.mask.pixels_in_mask
    )

    adapt_result = search.fit(model=adapt_model_pix, analysis=analysis)

    result.adapt = adapt_result

    return result
