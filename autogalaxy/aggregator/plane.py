from typing import List

import autofit as af

from autogalaxy.aggregator.abstract import AbstractAgg


def _plane_from(fit: af.Fit, galaxies: List["Galaxy"]) -> "Plane":
    """
    Returns a `Plane` object from a PyAutoFit database `Fit` object and an instance of galaxies from a non-linear
    search model-fit.

    This function adds the `hyper_model_image` and `hyper_galaxy_image_path_dict` to the galaxies before constructing
    the `Plane`, if they were used.

    Parameters
    ----------
    fit
        A PyAutoFit database Fit object containing the generators of the results of PyAutoGalaxy model-fits.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.

    Returns
    -------
    Plane
        The plane computed via an instance of galaxies.
    """

    from autogalaxy.galaxy.galaxy import Galaxy
    from autogalaxy.plane.plane import Plane

    hyper_model_image = fit.value(name="hyper_model_image")
    hyper_galaxy_image_path_dict = fit.value(name="hyper_galaxy_image_path_dict")

    galaxies_with_hyper = []

    if hyper_galaxy_image_path_dict is not None:

        galaxy_path_list = [
            gal[0] for gal in fit.instance.path_instance_tuples_for_class(Galaxy)
        ]

        for (galaxy_path, galaxy) in zip(galaxy_path_list, galaxies):

            if galaxy_path in hyper_galaxy_image_path_dict:
                galaxy.hyper_model_image = hyper_model_image
                galaxy.hyper_galaxy_image = hyper_galaxy_image_path_dict[galaxy_path]

            galaxies_with_hyper.append(galaxy)

        return Plane(galaxies=galaxies_with_hyper)

    return Plane(galaxies=galaxies)


class PlaneAgg(AbstractAgg):
    """
    Wraps a PyAutoFit aggregator in order to create generators of planes corresponding to the results of a non-linear
    search model-fit.
    """

    def make_object_for_gen(self, fit, galaxies) -> "Plane":
        """
        Creates a `Plane` object from a `ModelInstance` that contains the galaxies of a sample from a non-linear
        search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of PyAutoGalaxy model-fits.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.

        Returns
        -------
        Plane
            A plane whose galaxies are a sample of a PyAutoFit non-linear search.
        """
        return _plane_from(fit=fit, galaxies=galaxies)
