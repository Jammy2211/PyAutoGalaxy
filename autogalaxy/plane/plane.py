import numpy as np
from typing import Dict, Optional

import autoarray as aa

from autoarray.inversion.inversion.factory import inversion_imaging_unpacked_from
from autoarray.inversion.inversion.factory import inversion_interferometer_unpacked_from

from autogalaxy.lensing import LensingObject
from autogalaxy.profiles.light_profiles.light_profiles_snr import LightProfileSNR
from autogalaxy.galaxy.galaxy import Galaxy

from autogalaxy import exc
from autogalaxy.util import plane_util


class AbstractPlane(LensingObject):
    def __init__(
        self,
        galaxies,
        redshift: Optional[float] = None,
        profiling_dict: Optional[Dict] = None,
    ):
        """A plane of galaxies where all galaxies are at the same redshift.

        Parameters
        -----------
        redshift or None
            The redshift of the plane.
        galaxies : [Galaxy]
            The list of galaxies in this plane.
        """

        if redshift is None:

            if not galaxies:
                raise exc.PlaneException(
                    "No redshift and no galaxies were input to a Plane. A redshift for the Plane therefore cannot be"
                    "determined"
                )
            elif not all(
                [galaxies[0].redshift == galaxy.redshift for galaxy in galaxies]
            ):
                redshift = np.mean([galaxy.redshift for galaxy in galaxies])
            else:
                redshift = galaxies[0].redshift

        self.redshift = redshift
        self.galaxies = galaxies
        self.profiling_dict = profiling_dict

    @property
    def galaxy_redshifts(self):
        return [galaxy.redshift for galaxy in self.galaxies]

    @property
    def has_light_profile(self):
        if self.galaxies is not None:
            return any(
                list(map(lambda galaxy: galaxy.has_light_profile, self.galaxies))
            )

    @property
    def has_mass_profile(self):
        if self.galaxies is not None:
            return any(list(map(lambda galaxy: galaxy.has_mass_profile, self.galaxies)))

    @property
    def has_pixelization(self):
        return any([galaxy.pixelization for galaxy in self.galaxies])

    @property
    def has_regularization(self):
        return any([galaxy.regularization for galaxy in self.galaxies])

    @property
    def has_hyper_galaxy(self):
        return any(list(map(lambda galaxy: galaxy.has_hyper_galaxy, self.galaxies)))

    @property
    def galaxies_with_light_profile(self):
        return list(filter(lambda galaxy: galaxy.has_light_profile, self.galaxies))

    @property
    def galaxies_with_mass_profile(self):
        return list(filter(lambda galaxy: galaxy.has_mass_profile, self.galaxies))

    @property
    def galaxies_with_pixelization(self):
        return list(filter(lambda galaxy: galaxy.has_pixelization, self.galaxies))

    @property
    def galaxies_with_regularization(self):
        return list(filter(lambda galaxy: galaxy.has_regularization, self.galaxies))

    @property
    def pixelization_list(self):
        return [galaxy.pixelization for galaxy in self.galaxies_with_pixelization]

    @property
    def regularization_list(self):
        return [galaxy.regularization for galaxy in self.galaxies_with_pixelization]

    @property
    def hyper_galaxy_image_list(self):
        return [galaxy.hyper_galaxy_image for galaxy in self.galaxies_with_pixelization]

    @property
    def point_dict(self):

        point_dict = {}

        for galaxy in self.galaxies:
            for key, value in galaxy.point_dict.items():
                point_dict[key] = value

        return point_dict

    @property
    def mass_profiles(self):
        return [
            item
            for mass_profile in self.mass_profiles_of_galaxies
            for item in mass_profile
        ]

    @property
    def mass_profiles_of_galaxies(self):
        return [
            galaxy.mass_profiles for galaxy in self.galaxies if galaxy.has_mass_profile
        ]

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class in `Plane` as a `ValueIrregular` or `Grid2DIrregular` object.

        For example, if a plane has a galaxy which two light profiles and we want its axis-ratios, the following:

        `plane.extract_attribute(cls=LightProfile, name="axis_ratio")`

        would return:

        ValuesIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want their centres, the following:

        `plane.extract_attribute(cls=MassProfile, name="centres")`

        would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all mass profiles colored by their profile.
        """

        def extract(value, name):

            try:
                return getattr(value, name)
            except (AttributeError, IndexError):
                return None

        attributes = [
            extract(value, attr_name)
            for galaxy in self.galaxies
            for value in galaxy.__dict__.values()
            if isinstance(value, cls)
        ]

        if attributes == []:
            return None
        elif isinstance(attributes[0], float):
            return aa.ValuesIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return aa.Grid2DIrregular(grid=attributes)

    def extract_attributes_of_galaxies(self, cls, attr_name, filter_nones=False):
        """
        Returns an attribute of a class in the plane as a list of `ValueIrregular` or `Grid2DIrregular` objects,
        where the list indexes correspond to each galaxy in the plane..

        For example, if a plane has two galaxies which each have a light profile the following:

        `plane.extract_attributes_of_galaxies(cls=LightProfile, name="axis_ratio")`

        would return:

        [ValuesIrregular(values=[axis_ratio_0]), ValuesIrregular(values=[axis_ratio_1])]

        If a plane has two galaxies, the first with a mass profile and the second with two mass profiles ,the following:

        `plane.extract_attributes_of_galaxies(cls=MassProfile, name="centres")`

        would return:
        [
            Grid2DIrregular(grid=[(centre_y_0, centre_x_0)]),
            Grid2DIrregular(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1)])
        ]

        If a Profile does not have a certain entry, it is replaced with a None. Nones can be removed by
        setting `filter_nones=True`.

        This is used for visualization, for example plotting the centres of all mass profiles colored by their profile.
        """
        if filter_nones:

            return [
                galaxy.extract_attribute(cls=cls, attr_name=attr_name)
                for galaxy in self.galaxies
                if galaxy.extract_attribute(cls=cls, attr_name=attr_name) is not None
            ]

        else:

            return [
                galaxy.extract_attribute(cls=cls, attr_name=attr_name)
                for galaxy in self.galaxies
            ]


class AbstractPlaneLensing(AbstractPlane):
    @aa.grid_dec.grid_2d_to_structure
    def image_2d_from(self, grid):
        """
        Returns the profile-image plane image of the list of galaxies of the plane's sub-grid, by summing the
        individual images of each galaxy's light profile.

        The image is calculated on the sub-grid and binned-up to the original grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is `True`.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Parameters
        -----------

        """
        if self.galaxies:
            return sum(
                map(lambda galaxy: galaxy.image_2d_from(grid=grid), self.galaxies)
            )
        return np.zeros((grid.shape[0],))

    def images_of_galaxies_from(self, grid):
        return list(map(lambda galaxy: galaxy.image_2d_from(grid=grid), self.galaxies))

    def padded_image_2d_from(self, grid, psf_shape_2d):

        padded_grid = grid.padded_grid_from(kernel_shape_native=psf_shape_2d)

        return self.image_2d_from(grid=padded_grid)

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid):
        """
        Returns the convergence of the list of galaxies of the plane's sub-grid, by summing the individual convergences \
        of each galaxy's mass profile.

        The convergence is calculated on the sub-grid and binned-up to the original grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is `True`.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Parameters
        -----------
        grid : Grid2D
            The grid (or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
            potential is calculated on.
        galaxies : [Galaxy]
            The galaxies whose mass profiles are used to compute the surface densities.
        """
        if self.galaxies:
            return sum(map(lambda g: g.convergence_2d_from(grid=grid), self.galaxies))
        return np.zeros(shape=(grid.shape[0],))

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid):
        """
        Returns the potential of the list of galaxies of the plane's sub-grid, by summing the individual potentials \
        of each galaxy's mass profile.

        The potential is calculated on the sub-grid and binned-up to the original grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is `True`.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Parameters
        -----------
        grid : Grid2D
            The grid (or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
            potential is calculated on.
        galaxies : [Galaxy]
            The galaxies whose mass profiles are used to compute the surface densities.
        """
        if self.galaxies:
            return sum(map(lambda g: g.potential_2d_from(grid=grid), self.galaxies))
        return np.zeros((grid.shape[0]))

    @aa.grid_dec.grid_2d_to_structure
    def deflections_2d_from(self, grid):
        if self.galaxies:
            return sum(map(lambda g: g.deflections_2d_from(grid=grid), self.galaxies))
        return np.zeros(shape=(grid.shape[0], 2))

    @aa.grid_dec.grid_2d_to_structure
    def traced_grid_from(self, grid):
        """Trace this plane's grid_stacks to the next plane, using its deflection angles."""
        return grid - self.deflections_2d_from(grid=grid)


class AbstractPlaneData(AbstractPlaneLensing):
    def blurred_image_2d_via_psf_from(self, grid, psf, blurring_grid):

        image = self.image_2d_from(grid=grid)

        blurring_image = self.image_2d_from(grid=blurring_grid)

        return psf.convolved_array_with_mask_from(
            array=image.binned.native + blurring_image.binned.native, mask=grid.mask
        )

    def blurred_images_of_galaxies_via_psf_from(self, grid, psf, blurring_grid):
        return [
            galaxy.blurred_image_2d_via_psf_from(
                grid=grid, psf=psf, blurring_grid=blurring_grid
            )
            for galaxy in self.galaxies
        ]

    def blurred_image_2d_via_convolver_from(self, grid, convolver, blurring_grid):

        image = self.image_2d_from(grid=grid)

        blurring_image = self.image_2d_from(grid=blurring_grid)

        return convolver.convolve_image(image=image, blurring_image=blurring_image)

    def blurred_images_of_galaxies_via_convolver_from(
        self, grid, convolver, blurring_grid
    ):
        return [
            galaxy.blurred_image_2d_via_convolver_from(
                grid=grid, convolver=convolver, blurring_grid=blurring_grid
            )
            for galaxy in self.galaxies
        ]

    def unmasked_blurred_image_2d_via_psf_from(self, grid, psf):

        padded_grid = grid.padded_grid_from(kernel_shape_native=psf.shape_native)

        padded_image = self.image_2d_from(grid=padded_grid)

        return padded_grid.mask.unmasked_blurred_array_from(
            padded_array=padded_image, psf=psf, image_shape=grid.mask.shape
        )

    def unmasked_blurred_image_of_galaxies_via_psf_from(self, grid, psf):

        padded_grid = grid.padded_grid_from(kernel_shape_native=psf.shape_native)

        unmasked_blurred_images_of_galaxies = []

        for galaxy in self.galaxies:

            padded_image_1d = galaxy.image_2d_from(grid=padded_grid)

            unmasked_blurred_array_2d = padded_grid.mask.unmasked_blurred_array_from(
                padded_array=padded_image_1d, psf=psf, image_shape=grid.mask.shape
            )

            unmasked_blurred_images_of_galaxies.append(unmasked_blurred_array_2d)

        return unmasked_blurred_images_of_galaxies

    def profile_visibilities_via_transformer_from(self, grid, transformer):

        if self.galaxies:
            image = self.image_2d_from(grid=grid)
            return transformer.visibilities_from(image=image)
        else:
            return aa.Visibilities.zeros(
                shape_slim=(transformer.uv_wavelengths.shape[0],)
            )

    def profile_visibilities_of_galaxies_via_transformer_from(self, grid, transformer):
        return [
            galaxy.profile_visibilities_via_transformer_from(
                grid=grid, transformer=transformer
            )
            for galaxy in self.galaxies
        ]

    def sparse_image_plane_grid_list_from(
        self, grid, settings_pixelization=aa.SettingsPixelization()
    ):

        if not self.has_pixelization:
            return None

        return [
            pixelization.sparse_grid_from(
                grid=grid,
                hyper_image=hyper_galaxy_image,
                settings=settings_pixelization,
            )
            for pixelization, hyper_galaxy_image in zip(
                self.pixelization_list, self.hyper_galaxy_image_list
            )
        ]

    def mapper_from(
        self,
        grid,
        sparse_grid,
        pixelization,
        hyper_galaxy_image,
        sparse_image_plane_grid=None,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=aa.Preloads(),
    ):

        return pixelization.mapper_from(
            grid=grid,
            sparse_grid=sparse_grid,
            sparse_image_plane_grid=sparse_image_plane_grid,
            hyper_image=hyper_galaxy_image,
            settings=settings_pixelization,
            preloads=preloads,
            profiling_dict=self.profiling_dict,
        )

    def mapper_list_from(
        self,
        grid,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=aa.Preloads(),
    ):

        if not self.has_pixelization:
            return None

        sparse_grid_list = self.sparse_image_plane_grid_list_from(grid=grid)

        mapper_list = []

        pixelization_list = self.pixelization_list
        hyper_galaxy_image_list = self.hyper_galaxy_image_list

        for mapper_index in range(len(sparse_grid_list)):

            mapper = self.mapper_from(
                grid=grid,
                sparse_grid=sparse_grid_list,
                pixelization=pixelization_list[mapper_index],
                hyper_galaxy_image=hyper_galaxy_image_list[mapper_index],
                sparse_image_plane_grid=sparse_grid_list[mapper_index],
                settings_pixelization=settings_pixelization,
                preloads=preloads,
            )

            mapper_list.append(mapper)

        return mapper_list

    def inversion_imaging_from(
        self,
        grid,
        image,
        noise_map,
        convolver,
        w_tilde,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        preloads=aa.Preloads(),
    ):

        mapper_list = self.mapper_list_from(
            grid=grid, settings_pixelization=settings_pixelization, preloads=preloads
        )

        return inversion_imaging_unpacked_from(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            w_tilde=w_tilde,
            mapper_list=mapper_list,
            regularization_list=self.regularization_list,
            settings=settings_inversion,
            profiling_dict=self.profiling_dict,
        )

    def inversion_interferometer_from(
        self,
        grid,
        visibilities,
        noise_map,
        transformer,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        preloads=aa.Preloads(),
    ):

        mapper_list = self.mapper_list_from(
            grid=grid, settings_pixelization=settings_pixelization, preloads=preloads
        )

        return inversion_interferometer_unpacked_from(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            mapper_list=mapper_list,
            regularization_list=self.regularization_list,
            settings=settings_inversion,
            profiling_dict=self.profiling_dict,
        )

    def plane_image_2d_from(self, grid):
        return plane_util.plane_image_of_galaxies_from(
            shape=grid.mask.shape,
            grid=grid.mask.unmasked_grid_sub_1,
            galaxies=self.galaxies,
        )

    def hyper_noise_map_from(self, noise_map):
        hyper_noise_maps = self.hyper_noise_maps_of_galaxies_from(noise_map=noise_map)
        return sum(hyper_noise_maps)

    def hyper_noise_maps_of_galaxies_from(self, noise_map):
        """For a contribution map and noise-map, use the model hyper_galaxy galaxies to compute a hyper noise-map.

        Parameters
        -----------
        noise_map : imaging.NoiseMap or ndarray
            An arrays describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        """
        hyper_noise_maps = []

        for galaxy in self.galaxies:

            if galaxy.has_hyper_galaxy:

                hyper_noise_map_1d = galaxy.hyper_galaxy.hyper_noise_map_via_hyper_images_from(
                    noise_map=noise_map,
                    hyper_model_image=galaxy.hyper_model_image,
                    hyper_galaxy_image=galaxy.hyper_galaxy_image,
                )

                hyper_noise_maps.append(hyper_noise_map_1d)

            else:

                hyper_noise_map = aa.Array2D.manual_mask(
                    array=np.zeros(noise_map.mask.mask_sub_1.pixels_in_mask),
                    mask=noise_map.mask.mask_sub_1,
                )

                hyper_noise_maps.append(hyper_noise_map)

        return hyper_noise_maps

    @property
    def contribution_map(self):

        contribution_maps = self.contribution_maps_of_galaxies

        contribution_maps = [i for i in contribution_maps if i is not None]

        if contribution_maps:
            return sum(contribution_maps)
        else:
            return None

    @property
    def contribution_maps_of_galaxies(self):

        contribution_maps = []

        for galaxy in self.galaxies:

            if galaxy.hyper_galaxy is not None:

                contribution_maps.append(galaxy.contribution_map)

            else:

                contribution_maps.append(None)

        return contribution_maps

    def galaxy_image_dict_from(self, grid) -> {Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_image_dict = dict()

        images_of_galaxies = self.images_of_galaxies_from(grid=grid)
        for (galaxy_index, galaxy) in enumerate(self.galaxies):
            galaxy_image_dict[galaxy] = images_of_galaxies[galaxy_index]

        return galaxy_image_dict

    def galaxy_blurred_image_dict_via_convolver_from(
        self, grid, convolver, blurring_grid
    ) -> {Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_blurred_image_dict = dict()

        blurred_images_of_galaxies = self.blurred_images_of_galaxies_via_convolver_from(
            grid=grid, convolver=convolver, blurring_grid=blurring_grid
        )
        for (galaxy_index, galaxy) in enumerate(self.galaxies):
            galaxy_blurred_image_dict[galaxy] = blurred_images_of_galaxies[galaxy_index]

        return galaxy_blurred_image_dict

    def galaxy_profile_visibilities_dict_via_transformer_from(
        self, grid, transformer
    ) -> {Galaxy: np.ndarray}:
        """
        A dictionary associating galaxies with their corresponding model images
        """

        galaxy_profile_visibilities_image_dict = dict()

        profile_visibilities_of_galaxies = self.profile_visibilities_of_galaxies_via_transformer_from(
            grid=grid, transformer=transformer
        )
        for (galaxy_index, galaxy) in enumerate(self.galaxies):
            galaxy_profile_visibilities_image_dict[
                galaxy
            ] = profile_visibilities_of_galaxies[galaxy_index]

        return galaxy_profile_visibilities_image_dict

    def set_snr_of_snr_light_profiles(
        self,
        grid: aa.type.Grid2DLike,
        exposure_time: float,
        background_sky_level: float = 0.0,
    ):

        for galaxy in self.galaxies:
            for light_profile in galaxy.light_profiles:
                if isinstance(light_profile, LightProfileSNR):
                    light_profile.set_intensity_from(
                        grid=grid,
                        exposure_time=exposure_time,
                        background_sky_level=background_sky_level,
                    )


class Plane(AbstractPlaneData):

    pass


class PlaneImage:
    def __init__(self, array, grid):

        self.array = array
        self.grid = grid
