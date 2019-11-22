from autoarray.fit import fit


class GalaxyFit(fit.DatasetFit):
    def __init__(self, galaxy_data, model_galaxies):
        """Class which fits a set of galaxy-datas to a model galaxy, using either the galaxy's image, \
        surface-density or potential.

        Parameters
        ----------
        galaxy_data : GalaxyData
            The galaxy-datas object being fitted.
        model_galaxies : aast.Galaxy
            The model galaxy used to fit the galaxy-datas.
        """

        self.galaxy_data = galaxy_data
        self.model_galaxies = model_galaxies

        model_data = galaxy_data.profile_quantity_from_galaxies(galaxies=model_galaxies)

        super(GalaxyFit, self).__init__(
            data=galaxy_data.image,
            noise_map=galaxy_data.noise_map,
            mask=galaxy_data.mask,
            model_data=model_data.in_1d_binned,
        )

    @property
    def grid(self):
        return self.galaxy_data.grid

    def image(self):
        return self.data

    def model_image(self):
        return self.model_data

    @property
    def figure_of_merit(self):
        return self.likelihood
