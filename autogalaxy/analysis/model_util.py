from __future__ import annotations
import logging

import autofit as af
import autoarray as aa

from autogalaxy import exc

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")

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
        model_classes=(
            aa.AbstractImageMesh,
            aa.AbstractMesh,
            aa.AbstractRegularization,
        ),
        excluded_classes=(aa.reg.ConstantZeroth, aa.reg.Zeroth),
    )

    has_pix = model.has_model(
        cls=(aa.Pixelization,)
    ) or model.has_instance(cls=(aa.Pixelization,))

    if not has_pix:
        return None

    if pixelization_overwrite:
        model.galaxies.source.pixelization = af.Model(pixelization_overwrite)

    if regularization_overwrite:
        model.galaxies.source.regularization = af.Model(regularization_overwrite)

    if setup_adapt.mesh_pixels_fixed is not None:
        if hasattr(model.galaxies.source.pixelization.image_mesh, "pixels"):
            model.galaxies.source.pixelization.image_mesh.pixels = (
                setup_adapt.mesh_pixels_fixed
            )

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

    if analysis.adapt_images is None:
        raise exc.AnalysisException(
            "The analysis class adapt_images attribute is None, an instance of AdaptImages is required for adaptive fitting."
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

    analysis.adapt_images = result.adapt_images

    adapt_result = search.fit(model=adapt_model_pix, analysis=analysis)

    result.adapt = adapt_result

    return result
