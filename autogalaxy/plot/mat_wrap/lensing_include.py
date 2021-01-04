import copy

from autoconf import conf
from autoarray.plot.mat_wrap import include as inc
from autogalaxy import lensing

import typing


class Include1D(inc.Include1D):

    pass


class Include2D(inc.Include2D):
    def __init__(
        self,
        origin=None,
        mask=None,
        border=None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        critical_curves=None,
        caustics=None,
        multiple_images=None,
        mapper_source_pixelization_grid: typing.Optional[bool] = None,
        mapper_source_full_grid: typing.Optional[bool] = None,
        mapper_source_border: typing.Optional[bool] = None,
        mapper_data_pixelization_grid=None,
        preloaded_critical_curves=None,
        preload_caustics=None,
    ):

        super(Include2D, self).__init__(
            origin=origin,
            mask=mask,
            border=border,
            mapper_source_pixelization_grid=mapper_source_pixelization_grid,
            mapper_source_full_grid=mapper_source_full_grid,
            mapper_source_border=mapper_source_border,
            mapper_data_pixelization_grid=mapper_data_pixelization_grid,
        )

        self._positions = positions
        self._light_profile_centres = light_profile_centres
        self._mass_profile_centres = mass_profile_centres
        self._critical_curves = critical_curves
        self._caustics = caustics
        self._multiple_images = multiple_images

        self.preloaded_critical_curves = preloaded_critical_curves
        self.preloaded_caustics = preload_caustics

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

    def critical_curves_from_obj(self, obj):

        if not hasattr(obj, "has_mass_profile"):
            return None

        if not self.critical_curves or not obj.has_mass_profile:
            return None

        if self.preloaded_caustics is not None:
            return self.preloaded_critical_curves

        if isinstance(obj, lensing.LensingObject):
            try:
                return obj.critical_curves
            except Exception:
                print(
                    "Critical curve could not be calculated due to an unphysical mass model"
                )

    def caustics_from_obj(self, obj):

        if not hasattr(obj, "has_mass_profile"):
            return None

        if not self.caustics or not obj.has_mass_profile:
            return None

        if self.preloaded_caustics is not None:
            return self.preloaded_caustics

        if isinstance(obj, lensing.LensingObject):

            try:
                return obj.caustics
            except Exception:
                print(
                    "Caustics could not be calculated due to an unphysical mass model"
                )

    def traced_grid_of_plane_from_fit_and_plane_index(self, fit, plane_index):

        if self.positions is True:
            return fit.tracer.traced_grids_of_planes_from_grid(grid=fit.figure_grid)[
                plane_index
            ]

    def positions_of_plane_from_fit_and_plane_index(self, fit, plane_index):

        if self.positions is True:
            positions = self.positions_from_fit(fit=fit)
            if positions is None:
                return None

            return fit.tracer.traced_grids_of_planes_from_grid(grid=positions)[
                plane_index
            ]

    def new_include_with_preloaded_critical_curves_and_caustics(
        self, preloaded_critical_curves, preloaded_caustics
    ):

        include_2d = copy.deepcopy(self)
        include_2d.preloaded_critical_curves = preloaded_critical_curves
        include_2d.preloaded_caustics = preloaded_caustics

        return include_2d
