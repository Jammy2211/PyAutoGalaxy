from functools import partial
from typing import Optional

import autofit as af
import autoarray as aa


def _imaging_from(fit: af.Fit, settings_dataset: Optional[aa.SettingsImaging] = None) -> aa.Imaging:
    """
    Returns a `Imaging` object from a `PyAutoFit` sqlite database `Fit` object.

    The results of a model-fit can be stored in an sqlite database, including the following attributes of the fit:

    - The imaging data as a .fits file (`dataset/data.fits`).
    - The noise-map as a .fits file (`dataset/noise_map.fits`).
    - The point spread function as a .fits file (`dataset/psf.fits`).
    - The settings of the `Imaging` data structure used in the fit (`dataset/settings.json`).
    - The mask used to mask the `Imaging` data structure in the fit (`dataset/mask.fits`).

    Each individual attribute can be loaded from the database via the `fit.value()` method.

    This method combines all of these attributes and returns a `Imaging` object, has the mask applied to the
    `Imaging` data structure and its settings updated to the values used by the model-fit.

    The settings can be overwritten by inputting a `settings_dataset` object, for example if you want to use a grid
    with a different level of sub-griding.

    Parameters
    ----------
    fit
        A `PyAutoFit` `Fit` object which contains the results of a model-fit as an entry in a sqlite database.
    settings_dataset
        Optionally overwrite the `SettingsImaging` of the `Imaging` object that is created from the fit.
    """

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

    return dataset.apply_mask(mask=mask)


class ImagingAgg:
    def __init__(self, aggregator: af.Aggregator):
        self.aggregator = aggregator

    def dataset_gen_from(self, settings_dataset: Optional[aa.SettingsImaging] = None):
        """
        Returns a generator of `Imaging` objects from an input aggregator, which generates a list of the
        `Imaging` objects for every set of results loaded in the aggregator.

        This is performed by mapping the `imaging_from_agg_obj` with the aggregator, which sets up each
        imaging using only generators ensuring that manipulating the imaging of large sets of results is done in a
        memory efficient way.

        Parameters
        ----------
        aggregator
            A PyAutoFit aggregator object containing the results of model-fits.
        """

        func = partial(_imaging_from, settings_dataset=settings_dataset)

        return self.aggregator.map(func=func)
