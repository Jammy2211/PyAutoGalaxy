from autoarray.plotters import plotters
from autoastro import lensing



class Include(plotters.Include):
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
        inversion_centres=None,
        inversion_grid=None,
        inversion_border=None,
        inversion_image_pixelization_grid=None,
    ):

        super(Include, self).__init__(
            origin=origin,
            mask=mask,
            grid=grid,
            border=border,
            inversion_centres=inversion_centres,
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

    def mass_profile_centres_from_obj(self, obj):

        if self.mass_profile_centres:
            return obj.mass_profile_centres
        else:
            return None

    def mass_profile_centres_of_galaxies_from_obj(self, obj):

        if self.mass_profile_centres:
            return obj.mass_profile_centres_of_galaxies
        else:
            return None

    def mass_profile_centres_of_planes_from_obj(self, obj):

        if self.mass_profile_centres:
            return obj.mass_profile_centres_of_planes
        else:
            return None

    def critical_curves_from_obj(self, obj):

        if not self.critical_curves:
            return None

        if isinstance(obj, lensing.LensingObject) and obj.has_mass_profile:
            try:
                return obj.critical_curves
            except ValueError:
                print(
                    "Critical curve could not be calculated due to an unphysical mass model"
                )
                return None
        else:
            return None

    def caustics_from_obj(self, obj):

        if not self.caustics:
            return None

        if isinstance(obj, lensing.LensingObject) and obj.has_mass_profile:

            try:
                return obj.caustics
            except ValueError:
                print(
                    "Caustics could not be calculated due to an unphysical mass model"
                )
                return None
        else:

            return None

    def positions_from_fit(self, fit):
        """Get the masks of the fit if the masks should be plotted on the fit.

        Parameters
        -----------
        fit : datas.fitting.fitting.AbstractLensHyperFit
            The fit to the datas, which includes a lisrt of every model image, residual_map, chi-squareds, etc.
        mask : bool
            If *True*, the masks is plotted on the fit's datas.
        """
        if self.positions:
            return fit.masked_dataset.positions
        else:
            return None

    def inversion_image_pixelization_grid_from_fit(self, fit):

        if fit.inversion is not None:
            if self.inversion_image_pixelization_grid:
                if fit.inversion.mapper.is_image_plane_pixelization:
                    return fit.tracer.sparse_image_plane_grids_of_planes_from_grid(
                        grid=fit.grid
                    )[-1]

        return None
