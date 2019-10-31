import autoastro as aast
import numpy as np

# noinspection PyUnusedLocal
class MockLightProfile(aast.lp.LightProfile):
    def __init__(self, value, size=1):
        self.value = value
        self.size = size

    def profile_image_from_grid(self, grid):
        return np.array(self.size * [self.value])


class MockMassProfile(object):
    def __init__(self, value):
        self.value = value

    def surface_density_from_grid(self, grid):
        return np.array([self.value])

    def potential_from_grid(self, grid):
        return np.array([self.value])

    def deflections_from_grid(self, grid):
        return np.array([self.value, self.value])
