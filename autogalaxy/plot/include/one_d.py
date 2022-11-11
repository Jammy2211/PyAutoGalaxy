import autoarray.plot as aplt


class Include1D(aplt.Include1D):
    def __init__(self, half_light_radius=None, einstein_radius=None):

        super().__init__()

        self._half_light_radius = half_light_radius
        self._einstein_radius = einstein_radius

    @property
    def half_light_radius(self):
        return self.load(value=self._half_light_radius, name="half_light_radius")

    @property
    def einstein_radius(self):
        return self.load(value=self._einstein_radius, name="einstein_radius")
