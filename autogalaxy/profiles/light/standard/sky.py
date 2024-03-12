import numpy as np

import autoarray as aa



class Sky:

    def __init__(
        self,
        intensity: float = 0.1,
    ):
        """
        The sky light profile, representing the background sky emission as a constant sheet of values.

        Parameters
        ----------
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        """
        self.centre = (0.0, 0.0)
        self.ell_comps = (0.0, 0.0)

        self.intensity = intensity

    @aa.grid_dec.grid_2d_to_structure
    def image_2d_from(self, grid: aa.type.Grid2DLike):
        return np.full(shape=grid.shape[0], fill_value=self.intensity)


