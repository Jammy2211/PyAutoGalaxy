from itertools import count

import numpy as np

import autoarray as aa


class HyperGalaxy:
    _ids = count()

    def __init__(
        self,
        contribution_factor: float = 0.0,
        noise_factor: float = 0.0,
        noise_power: float = 1.0,
    ):
        """
        By using a ``HyperGalaxy``, the noise-map value in the regions of the image that the galaxy is located
        are increased.

        This prevents over-fitting regions of the data where the model does not provide a good fit
        (e.g. where a high chi-squared is inferred).

        This uses the parent ``Galaxy``'s `'contribution_map`', which determines the fraction of flux in every pixel
        of the image that is associated with that galaxy.

        The ``HyperGalaxy`` class contains free parameters which using the ``contribution_map`` then increase
        the noise.

        Using ``HyperGalaxy``'s to perform noise-scaling is described fully in the following ``HowToGalaxy``
        and ``HowToLens`` chapters:

        - https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/chapter_optional.html
        - https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_5_hyper_mode.html

        Parameters
        ----------
        contribution_factor
            Factor that adjusts how much of the galaxy's light is attributed to the
            contribution map.
        noise_factor
            Factor by which the noise-map is increased in the regions of the galaxy's
            contribution map.
        noise_power
            The power to which the contribution map is raised when scaling the
            noise-map.
        """
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

        self.component_number = next(self._ids)

    def __eq__(self, other):
        return (
            isinstance(other, HyperGalaxy)
            and self.contribution_factor == other.contribution_factor
            and self.noise_factor == other.noise_factor
            and self.noise_power == other.noise_power
        )

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

    def contribution_map_from(
        self, hyper_model_image: aa.Array2D, hyper_galaxy_image: aa.Array2D
    ) -> aa.Array2D:
        """
        Returns the contribution map of a galaxy, which represents the fraction of
        flux in each pixel that the galaxy is attributed to contain. This uses the
        `contribution_factor` free parameter.

        This is computed by dividing that galaxy's flux by the total flux in that
        pixel and then scaling by the maximum flux such that the contribution map
        ranges between 0 and 1.

        The contribution map is described in full in the following ``HowToGalaxy``
        and ``HowToLens`` chapters:

        - https://pyautogalaxy.readthedocs.io/en/latest/howtogalaxy/chapter_optional.html
        - https://pyautolens.readthedocs.io/en/latest/howtolens/chapter_5_hyper_mode.html

        Parameters
        ----------
        hyper_model_image
            The best-fit model image to the observed image from a previous analysis
            search. This provides the total light attributed to each image pixel by the
            model.
        hyper_galaxy_image
            A model image of the galaxy (from light profiles or an inversion) from a
            previous analysis search.

        Examples
        --------

        .. code-block:: python

            import autogalaxy as ag

            # For realistic use, input accurate image from a model-fit to all of the data.
            hyper_model_image = ag.Array2D.ones(shape_native=(100, 100), pixel_scales=1.0)

            # For realistic use, input accurate image from a model-fit to just one galaxy in the data.
            hyper_galaxu_image= ag.Array2D.ones(shape_native=(100, 100), pixel_scales=1.0)

            hyper_galaxy = ag.HyperGalaxy(contribution_factor=1.0)

            galaxy = ag.Galaxy(
                redshift=1.0,
                hyper_galaxy=hyper_galaxy,
            )

            contribution_map = galaxy.hyper_galaxy.contribution_map_from(
                hyper_galaxy_image=hyper_galaxu_image,
                hyper_model_image=hyper_model_image,
            )
        """
        contribution_map = np.divide(
            hyper_galaxy_image, np.add(hyper_model_image, self.contribution_factor)
        )
        return np.divide(contribution_map, np.max(contribution_map))

    def hyper_noise_map_from(
        self, noise_map: aa.Array2D, contribution_map: aa.Array2D
    ) -> aa.Array2D:
        """
        Returns a hyper noise-map from an input noise-map, which is a noise-map where certain noise map values
        (typing those corresponding to a poor fit and high chi-squared) are increased.

        Parameters
        ----------
        noise_map
            The input noise-map (before scaling), which normally corresponds to the `noise_map` of the data being
            fitted.
        contribution_map
            The contribution map of the galaxy, which represents the fraction of flux in each pixel that the galaxy
            is attributed to contain.

        Examples
        --------

        .. code-block:: python

            import autogalaxy as ag

            # For realistic use, input noise-map of observed data.
            hyper_model_image = ag.Array2D.ones(shape_native=(100, 100), pixel_scales=1.0)

            # For realistic use, input accurate image of a model-fit to all galaxies in the data.
            hyper_model_image = ag.Array2D.ones(shape_native=(100, 100), pixel_scales=1.0)

            # For realistic use, input accurate image from a model-fit of only this galaxy.
            hyper_galaxy_image = ag.Array2D.ones(shape_native=(100, 100), pixel_scales=1.0)

            hyper_galaxy = ag.HyperGalaxy(contribution_factor=1.0)

            galaxy = ag.Galaxy(
                redshift=1.0,
                hyper_galaxy=hyper_galaxy,
            )

            contribution_map = galaxy.hyper_galaxy.contribution_map_from(
                hyper_galaxy_image=hyper_galaxu_image,
                hyper_model_image=hyper_model_image,
            )

            hyper_noise_map = galaxy.hyper_galaxy.hyper_noise_map_from(
                noise_map=noise_map,
                contribution_map=contribution_map
            )
        """
        return self.noise_factor * (noise_map * contribution_map) ** self.noise_power
