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
        strong lens) estiamtes via an earlier model-fit. This class contains all adapt-images, and passes them
        around the source-code for using these adaptive schemes.

        Parameters
        ----------
        galaxy_image_dict
            A dictionary associating the name of each galaxy to an image of only that galaxy (e.g. for a strong lens
            the `source` entry is an image of the lensed source, without the lens light).
        """

        self.galaxy_image_dict = galaxy_image_dict
        self.galaxy_name_image_dict = galaxy_name_image_dict

    @property
    def mask(self):
        try:
            return list(self.galaxy_image_dict.values())[0].mask
        except AttributeError:
            return list(self.galaxy_name_image_dict.values())[0].mask

    @cached_property
    def model_image(self) -> aa.Array2D:
        """
        The model-image is the sum of all individual galaxy images in the image dictionary.
        """
        adapt_model_image = aa.Array2D(
            values=np.zeros(self.mask.derive_mask.sub_1.pixels_in_mask),
            mask=self.mask.derive_mask.sub_1,
        )

        for path in self.galaxy_image_dict.keys():
            adapt_model_image += self.galaxy_image_dict[path]

        return adapt_model_image

    @classmethod
    def from_result(cls, result) -> "AdaptImages":

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
