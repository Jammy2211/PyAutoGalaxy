from typing import Optional

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


class Include2D(aplt.Include2D):
    def __init__(
        self,
        origin=None,
        mask=None,
        border=None,
        grid=None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        critical_curves=None,
        caustics=None,
        multiple_images=None,
        mapper_source_pixelization_grid: Optional[bool] = None,
        mapper_source_grid_slim: Optional[bool] = None,
        mapper_data_pixelization_grid=None,
    ):

        super().__init__(
            origin=origin,
            mask=mask,
            border=border,
            grid=grid,
            mapper_source_pixelization_grid=mapper_source_pixelization_grid,
            mapper_source_grid_slim=mapper_source_grid_slim,
            mapper_data_pixelization_grid=mapper_data_pixelization_grid,
        )

        self._positions = positions
        self._light_profile_centres = light_profile_centres
        self._mass_profile_centres = mass_profile_centres
        self._critical_curves = critical_curves
        self._caustics = caustics
        self._multiple_images = multiple_images

    @property
    def positions(self):
        return self.load(value=self._positions, name="positions")

    @property
    def light_profile_centres(self):
        return self.load(
            value=self._light_profile_centres, name="light_profile_centres"
        )

    @property
    def mass_profile_centres(self):
        return self.load(value=self._mass_profile_centres, name="mass_profile_centres")

    @property
    def critical_curves(self):
        return self.load(value=self._critical_curves, name="critical_curves")

    @property
    def caustics(self):
        return self.load(value=self._caustics, name="caustics")

    @property
    def multiple_images(self):
        return self.load(value=self._multiple_images, name="multiple_images")
