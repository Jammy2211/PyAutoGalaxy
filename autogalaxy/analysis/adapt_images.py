from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional

import autoarray as aa

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy


class AdaptImages:
    def __init__(
        self,
        model_image: aa.Array2D,
        galaxy_image_dict: Optional[Dict[Galaxy, aa.Array2D]] = None,
        galaxy_name_image_dict: Optional[Dict[str, aa.Array2D]] = None,
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
        model_image
            The overall image of the galaxies or strong lens (e.g. lens and source) used by these adaptive schemes.
        galaxy_image_dict
            A dictionary associating the name of each galaxy to an image of only that galaxy (e.g. for a strong lens
            the `source` entry is an image of the lensed source, without the lens light).
        """

        self.model_image = model_image
        self.galaxy_image_dict = galaxy_image_dict
        self.galaxy_name_image_dict = galaxy_name_image_dict

    def updated_via_instance_from(self, instance) -> "AdaptImages":
        from autogalaxy.galaxy.galaxy import Galaxy

        galaxy_image_dict = {}

        for galaxy_name, galaxy in instance.path_instance_tuples_for_class(Galaxy):
            if galaxy_name in self.galaxy_name_image_dict:
                galaxy_image_dict[galaxy] = self.galaxy_name_image_dict[galaxy_name]

        return AdaptImages(
            model_image=self.model_image, galaxy_image_dict=galaxy_image_dict
        )
