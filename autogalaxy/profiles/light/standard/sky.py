import numpy as np
from typing import Optional

import autoarray as aa


class Sky:
    def __init__(
        self,
        level: float = 0.1,
    ):


        super().__init__(centre=None, ell_comps=None)

        self.intensity = intensity

    @aa.grid_dec.to_array
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None, **kwargs
    ):
        return np.full(shape=grid.shape[0], fill_value=self.intensity)
