from __future__ import annotations
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy
    from autogalaxy.plane.plane import Plane

import autofit as af
import autoarray as aa

from autogalaxy.aggregator.abstract import AbstractAgg


def galaxies_with_adapt_images_from(fit: af.Fit, galaxies: List[Galaxy]):
    from autogalaxy.galaxy.galaxy import Galaxy

    adapt_model_image = fit.value(name="adapt.adapt_model_image")

    if adapt_model_image is None:
        return galaxies

    adapt_model_image = aa.Array2D.from_primary_hdu(
        primary_hdu=adapt_model_image
    )

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
    Returns a `Plane` object from a PyAutoFit database `Fit` object and an instance of galaxies from a non-linear
    search model-fit.

    This function adds the `adapt_model_image` and `adapt_galaxy_image_path_dict` to the galaxies before constructing
    the `Plane`, if they were used.

    Parameters
    ----------
    fit
        A PyAutoFit database Fit object containing the generators of the results of model-fits.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.

    Returns
    -------
    Plane
        The plane computed via an instance of galaxies.
    """
    from autogalaxy.plane.plane import Plane

    galaxies = galaxies_with_adapt_images_from(fit=fit, galaxies=galaxies)

    return Plane(galaxies=galaxies)


class PlaneAgg(AbstractAgg):
    """
    Wraps a PyAutoFit aggregator in order to create generators of planes corresponding to the results of a non-linear
    search model-fit.
    """

    def object_via_gen_from(self, fit, galaxies) -> Plane:
        """
        Creates a `Plane` object from a `ModelInstance` that contains the galaxies of a sample from a non-linear
        search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of model-fits.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.

        Returns
        -------
        Plane
            A plane whose galaxies are a sample of a PyAutoFit non-linear search.
        """
        return _plane_from(fit=fit, galaxies=galaxies)
