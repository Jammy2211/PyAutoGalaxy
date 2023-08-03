from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy
    from autogalaxy.plane.plane import Plane

import autofit as af
import autoarray as aa

from autogalaxy.aggregator.abstract import AbstractAgg


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

    # TODO : Understand why fit.instance.path_instance_tuples_for_class(Galaxy) does not work because
    # TODO : it is a Plane opbject.
    # TODO : The code below used to be fit.instance.path_instance_tuples_for_class(Galaxy)

    instance = fit.model.instance_from_prior_medians(ignore_prior_limits=True)
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


def _plane_from(fit: af.Fit, galaxies: List[Galaxy]) -> Plane:
    """
    Returns an `Plane` object from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).
    - The adapt images associated with adaptive galaxy features (`adapt` folder).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `Plane` object for a given non-linear search sample
    (e.g. the maximum likelihood model). This includes associating adapt images with their respective galaxies.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.
    """

    from autogalaxy.plane.plane import Plane

    galaxies = galaxies_with_adapt_images_from(fit=fit, galaxies=galaxies)

    return Plane(galaxies=galaxies)


class PlaneAgg(AbstractAgg):
    """
    Interfaces with an `PyAutoFit` aggregator object to create instances of `Plane` objects from the results
    of a model-fit.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).
    - The adapt images associated with adaptive galaxy features (`adapt` folder).

    The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
    can load them all at once and create an `Plane` object via the `_plane_from` method.

    This class's methods returns generators which create the instances of the `Plane` objects. This ensures
    that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
    `Plane` instances in the memory at once.

    For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
    creates instances of the corresponding 3 `Plane` objects.

    This can be done manually, but this object provides a more concise API.

    Parameters
    ----------
    aggregator
        A `PyAutoFit` aggregator object which can load the results of model-fits.
    """

    def object_via_gen_from(self, fit, galaxies) -> Plane:
        """
        Returns a generator of `Plane` objects from an input aggregator.

        See `__init__` for a description of how the `Plane` objects are created by this method.

        Parameters
        ----------
        fit
            A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.
        """

        return _plane_from(fit=fit, galaxies=galaxies)
