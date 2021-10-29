import autoarray as aa

from autogalaxy.quantity.dataset_quantity import DatasetQuantity


class FitQuantity(aa.FitDataset):
    def __init__(self, quantity_dataset: DatasetQuantity, model_func):
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
        quantity_dataset
            The quantity that is to be fitted, which has a noise-map associated it with for computing goodness-of-fit
            metrics.
        """
        self.model_func = model_func

        model_data = model_func(grid=quantity_dataset.grid)

        fit = aa.FitData(
            data=quantity_dataset.data,
            noise_map=quantity_dataset.noise_map,
            model_data=model_data.binned,
            mask=quantity_dataset.mask,
            use_mask_in_fit=False,
        )

        super().__init__(dataset=quantity_dataset, fit=fit)

    @property
    def quantity_dataset(self):
        return self.dataset

    @property
    def grid(self):
        return self.quantity_dataset.grid
