from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from autoconf import conf
from autoconf import cached_property

import autoarray as aa

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy


class AdaptImages:
    def __init__(
        self,
        galaxy_image_dict: Optional[Dict[Galaxy, aa.Array2D]] = None,
        galaxy_name_image_dict: Optional[Dict[Tuple[str, ...], aa.Array2D]] = None,
    ):
        """
        Contains the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
        reconstructed galaxy's morphology.

        Pixelization image-mesh objects (e.g. `KMeans`, `Hilbert`) adapt the distribution of pixels to the observed
        image's brightness and therefore to the reconstructed source's morphology.

        Certain regularization schemes (e.g. `AdaptiveBrightness`) adapt their regularization coefficients to the
        reconstructed source's morphology.

        These adaptive schemes use "adapt-images", which are images of each galaxy (e.g. the lens and source of a
        strong lens) estimated via an earlier model-fit.

        The adapt-images are stored as the model-image of each galaxy in a model (e.g. the lens and source for a
        strong lens). They are stored as a dictionary mapping each instance of the galaxy to its model-image.

        For model-fitting, the galaxy instances are updated for every iteration of the non-linear search. This means
        an `AdaptImages` instance cannot be passed directly to an `Analysis` class, as the galaxy instances need to be
        updated for every iteration of the non-linear search.

        A dictionary mapping the path name of each galaxy (e.g. "galaxies.lens") to its model-image is therefore used
        which is called inside the `log_likelihood_function` o map the model-image of each galaxy to the galaxy
        instance of that iteration's specific model.

        Parameters
        ----------
        galaxy_image_dict
            A dictionary associating each galaxy instance to an image of only that galaxy (e.g. for a strong lens
            one entry will map an instance of the source galaxy entry to an image of the lensed source.
        galaxy_name_image_dict
            A dictionary associating each galaxy path name (e.g. "galaxies.source") to an image of only that
            galaxy (e.g. for a strong lens the `source` entry is an image of the lensed source, without the lens light).
        """

        self.galaxy_image_dict = galaxy_image_dict
        self.galaxy_name_image_dict = galaxy_name_image_dict

    @property
    def mask(self) -> aa.Mask2D:
        """
        The mask of the adapt images.
        """
        try:
            return list(self.galaxy_image_dict.values())[0].mask
        except AttributeError:
            return list(self.galaxy_name_image_dict.values())[0].mask

    @cached_property
    def model_image(self) -> aa.Array2D:
        """
        The model-image is the sum of all individual galaxy images in the image dictionary.

        This is computed by summing the model-image of each individual adapt galaxy contained in the dictionary.
        """
        adapt_model_image = aa.Array2D(
            values=np.zeros(self.mask.derive_mask.sub_1.pixels_in_mask),
            mask=self.mask.derive_mask.sub_1,
        )

        try:
            for path in self.galaxy_image_dict.keys():
                adapt_model_image += self.galaxy_image_dict[path]
        except AttributeError:
            for path in self.galaxy_name_image_dict.keys():
                adapt_model_image += self.galaxy_name_image_dict[path]

        return adapt_model_image

    @classmethod
    def from_result(cls, result) -> "AdaptImages":
        """
        Returns the adapt-images from a non-linear search result.

        For model-fitting, the adapt-images are typically setup using the maximum log likelihood model of the
        previous model-fit. This means the model-fitting is used to cleanly deblend the light of the different
        galaxies in the image (e.g. separate the lens light from the source light).

        This method uses attributes of a result (e.g. dictionary mapping galaxy instances to their model-images)
        to create the adapt-images.

        Certain models produce galaxy-images with negative flux values (e.g. a pixelization), which can cause
        numerical issues with the adaptive schemes. To prevent this, we set a minimum flux value for each
        galaxy-image, which is a fraction of the maximum flux value of that image defined via a config file.

        Parameters
        ----------
        result
            The result of a previous model-fit, which contains the model-image of each galaxy.

        Returns
        -------
        The adapt-images, which are the model-image of each galaxy inferred via the previous model-fit.
        """
        adapt_minimum_percent = conf.instance["general"]["adapt"][
            "adapt_minimum_percent"
        ]

        galaxy_name_image_dict = {}

        for path, galaxy in result.path_galaxy_tuples:
            galaxy_image = result.image_galaxy_dict[path]

            if not np.all(galaxy_image == 0):
                minimum_galaxy_value = adapt_minimum_percent * max(galaxy_image)
                galaxy_image[galaxy_image < minimum_galaxy_value] = minimum_galaxy_value

            galaxy_name_image_dict[path] = galaxy_image

        return AdaptImages(galaxy_name_image_dict=galaxy_name_image_dict)

    def updated_via_instance_from(self, instance, mask=None) -> "AdaptImages":
        """
        Returns adapt-images which have been updated to map galaxy instances instead of galaxy names.

        For model-fitting, the galaxy instances are updated for every iteration of the non-linear search. This means
        an `AdaptImages` instance cannot be passed directly to an `Analysis` class, as the galaxy instances need to be
        updated for every iteration of the non-linear search.

        A dictionary mapping the path name of each galaxy (e.g. "galaxies.lens") to its model-image is therefore used
        which is called inside the `log_likelihood_function` o map the model-image of each galaxy to the galaxy
        instance of that iteration's specific model.

        This function is also called when loading an `AdaptImages` instance from a PyAutoFit database, as the
        galaxy instances are also created on-fly from the database. Database images do not have a mask, so it is
        also applied to the adapt images on-the-fly during database loading.

        Parameters
        ----------
        instance
            The instance of the model-fit (e.g. in a non-linear search) which is used to update the adapt images.
        mask
            A mask which can be applied to the adapt images, which is used when setting up the adaptive images
            via the aggregator and autofit database tools.

        Returns
        -------

        """
        from autogalaxy.galaxy.galaxy import Galaxy

        galaxy_image_dict = {}

        for galaxy_name, galaxy in instance.path_instance_tuples_for_class(Galaxy):
            galaxy_name = str(galaxy_name)

            if galaxy_name in self.galaxy_name_image_dict:
                galaxy_image_dict[galaxy] = self.galaxy_name_image_dict[galaxy_name]

        if mask is not None:
            for key, image in galaxy_image_dict.items():
                galaxy_image_dict[key] = aa.Array2D(values=image, mask=mask)

        return AdaptImages(galaxy_image_dict=galaxy_image_dict)
