from typing import Dict

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy

class AdaptImages:

    def __init__(self, model_image : aa.Array2D, galaxy_image_path_dict : Dict[Galaxy, aa.Array2D]):
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
        galaxy_image_path_dict
            A dictionary associating the name of each galaxy to an image of only that galaxy (e.g. for a strong lens
            the `source` entry is an image of the lensed source, without the lens light).
        """

        self.model_image = model_image
        self.galaxy_image_path_dict = galaxy_image_path_dict