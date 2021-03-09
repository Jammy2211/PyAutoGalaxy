import numpy as np

import autofit as af
from autoarray.inversion import pixelizations as pix, regularization as reg
from autogalaxy.galaxy import galaxy as g

from typing import List, Optional


def isprior(obj):
    if isinstance(obj, af.PriorModel):
        return True
    return False


def isinstance_or_prior(obj, cls):
    if isinstance(obj, cls):
        return True
    if isinstance(obj, af.PriorModel) and obj.cls == cls:
        return True
    return False


def pixelization_from_model(model: af.CollectionPriorModel) -> pix.Pixelization:
    """
    For a model containing one or more galaxies, inspect its attributes and return the `pixelization` of a galaxy
    provided one galaxy has a pixelization, otherwise it returns none. There cannot be more than one `Pixelization` in
    a model.
    
    This function expects that the input model is a `CollectionPriorModel` where the first model-component has the
    name `galaxies`, and is itself a `CollectionPriorModel` of `Galaxy` and `GalaxyModel` instances. This is the
    standard API for creating a model in PyAutoGalaxy.

    The result of `pixelization_from_model` is used by the preloading to determine whether certain parts of a
    calculation can be cached before the non-linear search begins for efficiency.

    Parameters
    ----------
    model : af.CollectionPriorModel
        Contains the `galaxies` in the model that will be fitted via the non-linear search.

    Returns
    -------
    pix.Pixelization or None:
        The `Pixelization` of a galaxy, provided one galaxy has a `Pixelization`.
    """

    for galaxy in model.galaxies:
        if hasattr(galaxy, "pixelization"):
            if galaxy.pixelization is not None:
                if isinstance(galaxy.pixelization, af.PriorModel):
                    return galaxy.pixelization.cls
                else:
                    return galaxy.pixelization


def has_pixelization_from_model(model: af.CollectionPriorModel):
    """
    For a model containing one or more galaxies, inspect its attributes and return `True` if a galaxy has a
    `Pixelization` otherwise return `False`.

    This function expects that the input model is a `CollectionPriorModel` where the first model-component has the
    name `galaxies`, and is itself a `CollectionPriorModel` of `Galaxy` and `GalaxyModel` instances. This is the
    standard API for creating a model in PyAutoGalaxy.

    The result of `has_pixelization_from_model` is used by the preloading to determine whether certain parts of a
    calculation can be cached before the non-linear search begins for efficiency.

    Parameters
    ----------
    model : af.CollectionPriorModel
        Contains the `galaxies` in the model that will be fitted via the non-linear search.

    Returns
    -------
    pix.Pixelization or None:
        The `Pixelization` of a galaxy, provided one galaxy has a `Pixelization`.
    """
    pixelization = pixelization_from_model(model=model)

    return pixelization is not None


def pixelization_is_model_from_model(model: af.CollectionPriorModel):
    """
    For a model containing one or more galaxies, inspect its attributes and return `True` if a galaxy has a
    `Pixelization` which is a model-component with free parameters, otherwise return `False`. Therefore, a `False`
    may be returned if a galaxy has a `Pixelization` but it is an `instance` where no parameters are free parameters
    in the non-linear search.

    This function expects that the input model is a `CollectionPriorModel` where the first model-component has the
    name `galaxies`, and is itself a `CollectionPriorModel` of `Galaxy` and `GalaxyModel` instances. This is the
    standard API for creating a model in PyAutoGalaxy.

    The result of `pixelization_is_model_from_model` is used by the preloading to determine whether certain parts of a
    calculation can be cached before the non-linear search begins for efficiency.

    Parameters
    ----------
    model : af.CollectionPriorModel
        Contains the `galaxies` in the model that will be fitted via the non-linear search.

    Returns
    -------
    pix.Pixelization or None:
        The `Pixelization` of a galaxy, provided one galaxy has a `Pixelization`.
    """
    if model.galaxies:
        for galaxy in model.galaxies:
            if isprior(galaxy.pixelization):
                return True
    return False


def make_hyper_model_from(
    result: af.Result,
    hyper_galaxy_names: Optional[List[str]] = None,
    hyper_image_sky=None,
    hyper_background_noise=None,
) -> af.CollectionPriorModel:
    """
    Make a hyper model from the result of an `Analysis`, where a hyper-model corresponnds the maximum log likelihood
    instance of the inferred model but turns the following hyper model-components to free parameters:

    1) The `Pixelization` of any `Galaxy` in the model.
    2) The `Regularization` of any `Galaxy` in the model.
    3) Hyper data components like a `HyperImageSky` or `HyperBackgroundNoise` if input into the function.
    4) `HyperGalaxy` components of the `Galaxy`'s in the model, which are used to scale the noise in regions of the
    data which are fit poorly.

    The hyper model is typically used in pipelines to refine and improve an `Inversion` after model-fits that fit the
    `Galaxy` light and mass components.

    Parameters
    ----------
    result : af.Result
        The result of a previous `Analysis` phase whose maximum log likelihood model forms the basis of the hyper model.
    hyper_galaxy_names : [str]
        The names of the galaxies in the model which are used to create `HyperGalaxy` components that scale the
        noise-map.
    hyper_image_sky : hd.HyperImageSky
        The model used to optionally include the background sky of the data in the model.
    hyper_background_noise : hd.HyperBackgroundNoise
        The model used to optionally include the background noise of the data in the model.

    Returns
    -------
    af.CollectionPriorModel
        The hyper model, which has an instance of the input results maximum log likelihood model with certain hyper
        model components now free parameters.
    """

    model = result.instance.as_model((pix.Pixelization, reg.Regularization))

    if not has_pixelization_from_model(model=model):
        return None

    model.hyper_image_sky = hyper_image_sky
    model.hyper_background_noise = hyper_background_noise

    if hyper_galaxy_names is not None:

        for path_galaxy, galaxy in result.path_galaxy_tuples:
            if path_galaxy[-1] in hyper_galaxy_names:
                if not np.all(result.hyper_galaxy_image_path_dict[path_galaxy] == 0):

                    if "source" in path_galaxy[-1]:
                        setattr(
                            model.galaxies.source,
                            "hyper_galaxy",
                            af.PriorModel(g.HyperGalaxy),
                        )
                    elif "lens" in path_galaxy[-1]:
                        setattr(
                            model.galaxies.lens,
                            "hyper_galaxy",
                            af.PriorModel(g.HyperGalaxy),
                        )

    return model
