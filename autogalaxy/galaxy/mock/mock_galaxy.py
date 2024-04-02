import numpy as np

import autoarray as aa


class MockGalaxy:
    def __init__(self, value, shape=1):
        self.value = value
        self.shape = shape

    @aa.grid_dec.grid_2d_to_structure
    def image_2d_from(self, grid):
        return np.full(shape=self.shape, fill_value=self.value)

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid):
        return np.full(shape=self.shape, fill_value=self.value)

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid):
        return np.full(shape=self.shape, fill_value=self.value)

    @aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid):
        return np.full(shape=(self.shape, 2), fill_value=self.value)
