import autofit as af
from autoarray.inversion import pixelizations as pix
from autogalaxy.galaxy.galaxy_model import is_light_profile_class
from autogalaxy.pipeline.phase.abstract.result import Result
from astropy import cosmology as cosmo


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


def pixelization_from_model(model : af.CollectionPriorModel) -> pix.Pixelization:
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


def has_pixelization_from_model(model : af.CollectionPriorModel):
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


def pixelization_is_model_from_model(model : af.CollectionPriorModel):
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


