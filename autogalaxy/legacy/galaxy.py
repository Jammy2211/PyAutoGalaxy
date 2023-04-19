from typing import Optional

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy as GalaxyBase
from autogalaxy.legacy.hyper import HyperGalaxy


class Galaxy(GalaxyBase):
    """
    @DynamicAttrs
    """

    def __init__(
        self, redshift: float, hyper_galaxy: Optional[HyperGalaxy] = None, **kwargs
    ):
        """
        Class representing a galaxy, which is composed of attributes used for fitting hyper_galaxies (e.g. light profiles, \
        mass profiles, pixelizations, etc.).

        All *has_* methods retun `True` if galaxy has that attribute, `False` if not.

        Parameters
        ----------
        redshift
            The redshift of the galaxy.
        pixelization
            The pixelization of the galaxy used to reconstruct an observed image using an inversion.

        Attributes
        ----------
        adapt_model_image
            The best-fit model image to the observed image from a previous analysis
            search. This provides the total light attributed to each image pixel by the
            model.
        adapt_galaxy_image
            A model image of the galaxy (from light profiles or an inversion) from a
            previous analysis search.
        """
        super().__init__(redshift=redshift, **kwargs)

        self.hyper_galaxy = hyper_galaxy

    @property
    def contribution_map(self) -> aa.Array2D:
        """
        Returns the contribution map of a galaxy, which represents the fraction of
        flux in each pixel that the galaxy is attributed to contain, hyper to the
        *contribution_factor* hyper_galaxies-parameter.

        This is computed by dividing that galaxy's flux by the total flux in that \
        pixel and then scaling by the maximum flux such that the contribution map \
        ranges between 0 and 1.

        Parameters
        ----------

        """
        return self.hyper_galaxy.contribution_map_from(
            adapt_model_image=self.adapt_model_image,
            adapt_galaxy_image=self.adapt_galaxy_image,
        )
