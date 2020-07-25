from autoarray.dataset import abstract_dataset
from autoarray.structures import arrays, grids
from autogalaxy import exc


class MaskedGalaxyDataset:
    def __init__(
        self,
        galaxy_data,
        mask,
        grid_class=grids.Grid,
        fractional_accuracy=0.9999,
        sub_steps=None,
        pixel_scales_interp=None,
        use_image=False,
        use_convergence=False,
        use_potential=False,
        use_deflections_y=False,
        use_deflections_x=False,
    ):
        """ A galaxy-fit data is a collection of fit data components which are used to fit a galaxy to another galaxy. \
        This is where a component of a galaxy's light profiles (e.g. image) or mass profiles (e.g. surface \
        density, potential or deflection angles) are fitted to one another.

        This is primarily performed for automatic prior linking, as a means to efficiently link the priors of a galaxy \
        using one inferred parametrization of light or mass profiles to a new galaxy with a different parametrization \
        of light or mass profiles.

        This omits a number of the fit data components typically used when fitting an image (e.g. the observed image, PSF, \
        exposure time map), but still has a number of the other components (e.g. an effective noise_map, grid_stacks).

        Parameters
        ----------
        galaxy_data : GalaxyData
            The collection of data about the galaxy (image of its profile map, noise-map, etc.) that is fitted.
        mask: aa.AbstractMask
            The 2D masks that is applied to image fit data.
        sub_size : int
            The size of the sub-grid used for computing the SubGrid (see imaging.masks.SubGrid).

        Attributes
        ----------
        noise_map_1d : ndarray
            The masked 1D arrays of the noise_map
        grid_stacks : imaging.masks.GridStack
            Grids of (y,x) Cartesian coordinates which map over the masked 1D fit data arrays's pixels (includes an \
            grid, sub-grid, etc.)
        """
        self.mask = mask
        self.galaxy_data = galaxy_data
        self.pixel_scales = galaxy_data.pixel_scales

        self.image = arrays.Array.manual_mask(
            array=galaxy_data.image.in_2d_binned, mask=mask.mask_sub_1
        )
        self.noise_map = arrays.Array.manual_mask(
            array=galaxy_data.noise_map.in_2d_binned, mask=mask.mask_sub_1
        )

        self.signal_to_noise_map = self.image / self.noise_map

        self.sub_size = mask.sub_size

        self.grid = abstract_dataset.grid_from_mask_and_grid_class(
            mask=mask,
            grid_class=grid_class,
            fractional_accuracy=fractional_accuracy,
            sub_steps=sub_steps,
            pixel_scales_interp=pixel_scales_interp,
        )

        if all(
            not element
            for element in [
                use_image,
                use_convergence,
                use_potential,
                use_deflections_y,
                use_deflections_x,
            ]
        ):
            raise exc.GalaxyException(
                "The galaxy fit data has not been supplied with a use_ method."
            )

        if (
            sum(
                [
                    use_image,
                    use_convergence,
                    use_potential,
                    use_deflections_y,
                    use_deflections_x,
                ]
            )
            > 1
        ):
            raise exc.GalaxyException(
                "The galaxy fit data has not been supplied with multiple use_ methods, only supply "
                "one."
            )

        self.use_image = use_image
        self.use_convergence = use_convergence
        self.use_potential = use_potential
        self.use_deflections_y = use_deflections_y
        self.use_deflections_x = use_deflections_x

    def profile_quantity_from_galaxies(self, galaxies):

        if self.use_image:
            image = sum(map(lambda g: g.image_from_grid(grid=self.grid), galaxies))
            return arrays.Array.manual_mask(array=image, mask=self.mask)
        elif self.use_convergence:
            convergence = sum(
                map(lambda g: g.convergence_from_grid(grid=self.grid), galaxies)
            )
            return arrays.Array.manual_mask(array=convergence, mask=self.mask)
        elif self.use_potential:
            potential = sum(
                map(lambda g: g.potential_from_grid(grid=self.grid), galaxies)
            )
            return arrays.Array.manual_mask(array=potential, mask=self.mask)
        elif self.use_deflections_y:
            deflections = sum(
                map(lambda g: g.deflections_from_grid(grid=self.grid), galaxies)
            )
            return arrays.Array.manual_mask(
                array=deflections[:, 0], mask=self.grid.mask
            )
        elif self.use_deflections_x:
            deflections = sum(
                map(lambda g: g.deflections_from_grid(grid=self.grid), galaxies)
            )
            return arrays.Array.manual_mask(
                array=deflections[:, 1], mask=self.grid.mask
            )

    @property
    def data(self):
        return self.image
