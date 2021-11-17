from typing import Union

import autoarray as aa

from autogalaxy.quantity.dataset_quantity import DatasetQuantity
from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.profiles.mass_profiles.mass_profiles import MassProfile
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plane.plane import Plane


class FitQuantity(aa.FitDataset):
    def __init__(self, dataset: DatasetQuantity, light_mass_obj: Union[LightProfile, MassProfile, Galaxy, Plane], func_str: str):
        """
        Fits a `DatasetQuantity` object with model data.

        This is used to fit a quantity (e.g. a convergence, deflection angles), from an object like
        a `LightProfile`, `MassProfile`, `Galaxy` or `Plane`, to the same quantity derived from another of that object.

        For example, we may have the 2D convergence of a power-law mass profile and wish to determine how closely the
        2D convergence of an nfw mass profile's matches it. The `FitQuantity` can fit the two, where a noise-map
        is associated with the quantity's dataset such that figure of merits like a chi-squared and log likelihood
        can be computed.

        This is ultimately used in the `AnalysisQuantity` class to perform model-fitting of quantities of different
        mass profiles, light profiles, galaxies, etc.

        Parameters
        ----------
        dataset
            The quantity that is to be fitted, which has a noise-map associated it with for computing goodness-of-fit
            metrics.
        light_mass_obj
            An object containing functions which computes a light and / or mass quantity (e.g. a plane of galaxies)
            whose model quantities are used to fit the quantity data.
        func_str
            A string giving the name of the method of the input `Plane` used to compute the quantity that fits
            the dataset.          
        """

        self.light_mass_obj = light_mass_obj
        self.quantity_str = func_str

        func = getattr(light_mass_obj, func_str)

        model_data = func(grid=dataset.grid)

        fit = aa.FitData(
            data=dataset.data,
            noise_map=dataset.noise_map,
            model_data=model_data.binned,
            mask=dataset.mask,
            use_mask_in_fit=False,
        )

        super().__init__(dataset=dataset, fit=fit)

    @property
    def quantity_dataset(self):
        return self.dataset

    @property
    def mask(self):
        return self.fit.mask

    @property
    def grid(self):
        return self.quantity_dataset.grid

    @property
    def plane(self):
        return self.light_mass_obj