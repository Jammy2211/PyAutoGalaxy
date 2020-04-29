from autoarray.fit import fit


class FitGalaxy(fit.FitDataset):
    def __init__(self, masked_galaxy_dataset, model_galaxies):
        """Class which fits a set of galaxy-datas to a model galaxy, using either the galaxy's image, \
        surface-density or potential.

        Parameters
        ----------
        masked_galaxy_dataset : GalaxyData
            The galaxy-datas object being fitted.
        model_galaxies : ag.Galaxy
            The model galaxy used to fit the galaxy-datas.
        """
        self.model_galaxies = model_galaxies

        model_data = masked_galaxy_dataset.profile_quantity_from_galaxies(
            galaxies=model_galaxies
        )

        super(FitGalaxy, self).__init__(
            masked_dataset=masked_galaxy_dataset, model_data=model_data.in_1d_binned
        )

    @property
    def masked_galaxy_dataset(self):
        return self.masked_dataset

    @property
    def grid(self):
        return self.masked_galaxy_dataset.grid

    def image(self):
        return self.data

    def model_image(self):
        return self.model_data

    @property
    def figure_of_merit(self):
        return self.log_likelihood
