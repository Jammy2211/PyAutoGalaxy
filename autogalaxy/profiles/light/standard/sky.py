import numpy as np
from typing import Optional

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile


class Sky(LightProfile):
    def __init__(
        self,
        intensity: float = 0.1,
    ):
        """
        The sky light profile, representing the background sky emission as a constant sheet of values.

        To be consistent with other parts of the light profile API, the sky is passed a centre and elliptical
        components, but these are not used. The sky is a constant value across the whole image and therefore
        does not have an image which depends on these geometric parameters.

        Parameters
        ----------
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        """

        super().__init__(centre=None, ell_comps=None)

        self.intensity = intensity

    @aa.grid_dec.to_array
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None, **kwargs
    ):
        return np.full(shape=grid.shape[0], fill_value=self.intensity)
