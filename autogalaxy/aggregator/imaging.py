from functools import partial
from typing import List, Optional

import autofit as af
import autoarray as aa


def _imaging_from(
    fit: af.Fit, settings_dataset: Optional[aa.SettingsImaging] = None
) -> List[aa.Imaging]:
    """
    Returns a list of `Imaging` objects from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

    - The imaging data as a .fits file (`dataset/data.fits`).
    - The noise-map as a .fits file (`dataset/noise_map.fits`).
    - The point spread function as a .fits file (`dataset/psf.fits`).
    - The settings of the `Imaging` data structure used in the fit (`dataset/settings.json`).
    - The mask used to mask the `Imaging` data structure in the fit (`dataset/mask.fits`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `Imaging` object, has the mask applied to the
    `Imaging` data structure and its settings updated to the values used by the model-fit.

    If multiple `Imaging` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
    is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
    `Imaging` objects.

    The settings can be overwritten by inputting a `settings_dataset` object, for example if you want to use a grid
    with a different level of sub-griding.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    settings_dataset
        Optionally overwrite the `SettingsImaging` of the `Imaging` object that is created from the fit.
    """

    fit_list = [fit] if not fit.children else fit.children

    dataset_list = []

    for fit in fit_list:
        data = aa.Array2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.data"))
        noise_map = aa.Array2D.from_primary_hdu(
            primary_hdu=fit.value(name="dataset.noise_map")
        )
        psf = aa.Kernel2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.psf"))

        settings_dataset = settings_dataset or fit.value(name="dataset.settings")

        dataset = aa.Imaging(
            data=data,
            noise_map=noise_map,
            psf=psf,
            settings=settings_dataset,
            check_noise_map=False,
        )

        mask = aa.Mask2D.from_primary_hdu(primary_hdu=fit.value(name="dataset.mask"))

        dataset = dataset.apply_mask(mask=mask)

        dataset_list.append(dataset)

    return dataset_list


class ImagingAgg:
    def __init__(self, aggregator: af.Aggregator):
        """
        Interfaces with an `PyAutoFit` aggregator object to create instances of `Imaging` objects from the results
        of a model-fit.

        The results of a model-fit can be stored in a sqlite database, including the following attributes of the fit:

        - The imaging data as a .fits file (`dataset/data.fits`).
        - The noise-map as a .fits file (`dataset/noise_map.fits`).
        - The point spread function as a .fits file (`dataset/psf.fits`).
        - The settings of the `Imaging` data structure used in the fit (`dataset/settings.json`).
        - The mask used to mask the `Imaging` data structure in the fit (`dataset/mask.fits`).

        The `aggregator` contains the path to each of these files, and they can be loaded individually. This class
        can load them all at once and create an `Imaging` object via the `_imaging_from` method.

        This class's methods returns generators which create the instances of the `Imaging` objects. This ensures
        that large sets of results can be efficiently loaded from the hard-disk and do not require storing all
        `Imaging` instances in the memory at once.

        For example, if the `aggregator` contains 3 model-fits, this class can be used to create a generator which
        creates instances of the corresponding 3 `Imaging` objects.

        If multiple `Imaging` objects were fitted simultaneously via analysis summing, the `fit.child_values()` method
        is instead used to load lists of the data, noise-map, PSF and mask and combine them into a list of
        `Imaging` objects.

        This can be done manually, but this object provides a more concise API.

        Parameters
        ----------
        aggregator
            A `PyAutoFit` aggregator object which can load the results of model-fits.
        """
        self.aggregator = aggregator

    def dataset_gen_from(
        self, settings_dataset: Optional[aa.SettingsImaging] = None
    ) -> List[aa.Imaging]:
        """
        Returns a generator of `Imaging` objects from an input aggregator.

        See `__init__` for a description of how the `Imaging` objects are created by this method.

        The settings can be overwritten by inputting a `settings_dataset` object, for example if you want to use a grid
        with a different level of sub-griding.

        Parameters
        ----------
        settings_dataset
            Optionally overwrite the `SettingsImaging` of the `Imaging` object that is created from the fit.
        """

        func = partial(_imaging_from, settings_dataset=settings_dataset)

        return self.aggregator.map(func=func)
