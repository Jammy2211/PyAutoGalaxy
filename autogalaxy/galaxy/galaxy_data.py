class GalaxyData:
    def __init__(self, image, noise_map, pixel_scales):
        """ A galaxy-fit data is a collection of fit data components which are used to fit a galaxy to another galaxy. \
        This is where a component of a galaxy's light profiles (e.g. image) or mass profiles (e.g. convergence \
        , potential or deflection angles) are fitted to one another.

        This is primarily performed for automatic prior linking, as a means to efficiently link the priors of a galaxy \
        using one inferred parametrization of light or mass profiles to a new galaxy with a different parametrization \
        of light or mass profiles.

        This omits a number of the fit data components typically used when fitting an image (e.g. the observed image, PSF, \
        exposure time map), but still has a number of the other components (e.g. an effective noise_map, grid_stacks).

        Parameters
        ----------
        image : aa.Array
            An image of the quantity of the galaxy that is being fitted (e.g. its image, convergence, etc.).
        noise_map : aa.Scaled
            The noise_map used for computing the log likelihood of each fit. This can be chosen arbritarily.
        """
        self.image = image
        self.noise_map = noise_map
        self.pixel_scales = pixel_scales
