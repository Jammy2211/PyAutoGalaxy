import copy

from autoconf import conf
from autoarray.plot.mat_wrap import include as inc
from autogalaxy import lensing


class Include(inc.Include):
    def __init__(
        self,
        origin=None,
        mask=None,
        grid=None,
        border=None,
        positions=None,
        light_profile_centres=None,
        mass_profile_centres=None,
        critical_curves=None,
        caustics=None,
        multiple_images=None,
        inversion_pixelization_grid=None,
        inversion_grid=None,
        inversion_border=None,
        inversion_image_pixelization_grid=None,
        preloaded_critical_curves=None,
        preload_caustics=None,
    ):

        super(Include, self).__init__(
            origin=origin,
            mask=mask,
            grid=grid,
            border=border,
            inversion_pixelization_grid=inversion_pixelization_grid,
            inversion_grid=inversion_grid,
            inversion_border=inversion_border,
            inversion_image_pixelization_grid=inversion_image_pixelization_grid,
        )

        self.positions = self.load_include(value=positions, name="positions")
        self.light_profile_centres = self.load_include(
            value=light_profile_centres, name="light_profile_centres"
        )
        self.mass_profile_centres = self.load_include(
            value=mass_profile_centres, name="mass_profile_centres"
        )
        self.critical_curves = self.load_include(
            value=critical_curves, name="critical_curves"
        )

        self.caustics = self.load_include(value=caustics, name="caustics")
        self.multiple_images = self.load_include(
            value=multiple_images, name="multiple_images"
        )

        self.preloaded_critical_curves = preloaded_critical_curves
        self.preloaded_caustics = preload_caustics

    @staticmethod
    def load_include(value, name):
        if value is not None:
            return value
        return conf.instance["visualize"]["general"]["include"][name]

    def positions_from_masked_dataset(self, masked_dataset):

        if self.positions:
            return masked_dataset.positions

    def light_profile_centres_from_obj(self, obj):

        if self.light_profile_centres:
            return obj.light_profile_centres

    def mass_profile_centres_from_obj(self, obj):

        if self.mass_profile_centres:
            return obj.mass_profile_centres

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

    def positions_from_fit(self, fit):
        """Get the masks of the fit if the masks should be plotted on the fit.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
        mask : bool
            If `True`, the masks is plotted on the fit's datas.
        """
        if self.positions:
            try:
                return fit.settings_masked_dataset.positions
            except AttributeError:
                return None

    def traced_grid_of_plane_from_fit_and_plane_index(self, fit, plane_index):

        if self.positions is True:
            return fit.tracer.traced_grids_of_planes_from_grid(grid=fit.grid)[
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

    def inversion_image_pixelization_grid_from_fit(self, fit):

        if fit.inversion is not None:
            if self.inversion_image_pixelization_grid:
                if fit.inversion.mapper.is_image_plane_pixelization:
                    return fit.tracer.sparse_image_plane_grids_of_planes_from_grid(
                        grid=fit.grid
                    )[-1]

    def new_include_with_preloaded_critical_curves_and_caustics(
        self, preloaded_critical_curves, preloaded_caustics
    ):

        include = copy.deepcopy(self)
        include.preloaded_critical_curves = preloaded_critical_curves
        include.preloaded_caustics = preloaded_caustics

        return include
