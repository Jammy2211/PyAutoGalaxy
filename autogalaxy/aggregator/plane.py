from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy
    from autogalaxy.plane.plane import Plane

import autofit as af

from autogalaxy.aggregator.abstract import AbstractAgg

from autogalaxy.aggregator import agg_util

logger = logging.getLogger(__name__)


def _plane_from(fit: af.Fit, galaxies: List[Galaxy]) -> List[Plane]:
    """
    Returns an `Plane` object from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The model and its best fit parameters (e.g. `model.json`).
    - The adapt images associated with adaptive galaxy features (`adapt` folder).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `Plane` object for a given non-linear search sample
    (e.g. the maximum likelihood model). This includes associating adapt images with their respective galaxies.

    If multiple `Plane` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of planes. This is necessary if each plane has different galaxies (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    galaxies
        A list of galaxies corresponding to a sample of a non-linear search and model-fit.
    """

    from autogalaxy.plane.plane import Plane

    if len(fit.children) > 0:
        logger.info(
            """
            Using database for a fit with multiple summed Analysis objects.

            Plane objects do not fully support this yet (e.g. variables across Analysis objects may not be correct)
            so proceed with caution!
            """
        )

        return [Plane(galaxies=galaxies)] * len(fit.children)

    return [Plane(galaxies=galaxies)]


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

    If multiple `Plane` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of planes. This is necessary if each plane has different galaxies (e.g. certain
    parameters vary across each dataset and `Analysis` object).

    This can be done manually, but this object provides a more concise API.

    Parameters
    ----------
    aggregator
        A `PyAutoFit` aggregator object which can load the results of model-fits.
    """

    def object_via_gen_from(self, fit, galaxies) -> List[Plane]:
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
