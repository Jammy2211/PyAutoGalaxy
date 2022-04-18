import json
import numpy as np
from typing import Dict, List, Optional, Union

import autoarray as aa

from autoconf.dictable import Dictable

from autoarray.inversion.inversion.factory import inversion_imaging_unpacked_from
from autoarray.inversion.inversion.factory import inversion_interferometer_unpacked_from
from autogalaxy import exc
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.light_profiles.light_profiles_snr import LightProfileSNR
from autogalaxy.operate.image import OperateImageGalaxies
from autogalaxy.operate.deflections import OperateDeflections

from autogalaxy.util import plane_util


class Plane(OperateImageGalaxies, OperateDeflections, Dictable):
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

    def dict(self) -> Dict:
        plane_dict = super().dict()
        plane_dict["galaxies"] = [galaxy.dict() for galaxy in self.galaxies]
        return plane_dict

    def output_to_json(self, file_path: str):

        with open(file_path, "w+") as f:
            json.dump(self.dict(), f, indent=4)

    @property
    def galaxy_redshifts(self) -> List[float]:
        return [galaxy.redshift for galaxy in self.galaxies]

    @property
    def has_light_profile(self) -> bool:
        if self.galaxies is not None:
            return any(
                list(map(lambda galaxy: galaxy.has_light_profile, self.galaxies))
            )

    @aa.grid_dec.grid_2d_to_structure
    def image_2d_from(self, grid: aa.type.Grid2DLike) -> aa.Array2D:
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

    def image_2d_list_from(self, grid: aa.type.Grid2DLike) -> List[aa.Array2D]:
        return list(map(lambda galaxy: galaxy.image_2d_from(grid=grid), self.galaxies))

    def galaxy_image_2d_dict_from(
        self, grid: aa.type.Grid2DLike
    ) -> {Galaxy: np.ndarray}:
        """
        Returns a dictionary associating every `Galaxy` object in the `Plane` with its corresponding 2D image, using
        the instance of each galaxy as the dictionary keys.

        This object is used for hyper-features, which use the image of each galaxy in a model-fit in order to
        adapt quantities like a pixelization or regularization scheme to the surface brightness of the galaxies being
        fitted.

        By inheriting from `OperateImageGalaxies` functions which apply operations of this dictionary are accessible,
        for example convolving every image with a PSF or applying a Fourier transform to create a galaxy-visibilities
        dictionary.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.

        Returns
        -------
        A dictionary associated every galaxy in the plane with its corresponding 2D image.
        """

        galaxy_image_dict = dict()

        images_of_galaxies = self.image_2d_list_from(grid=grid)
        for (galaxy_index, galaxy) in enumerate(self.galaxies):
            galaxy_image_dict[galaxy] = images_of_galaxies[galaxy_index]

        return galaxy_image_dict

    def plane_image_2d_from(self, grid: aa.type.Grid2DLike) -> "PlaneImage":
        return plane_util.plane_image_of_galaxies_from(
            shape=grid.mask.shape,
            grid=grid.mask.unmasked_grid_sub_1,
            galaxies=self.galaxies,
        )

    @property
    def has_mass_profile(self) -> bool:
        if self.galaxies is not None:
            return any(list(map(lambda galaxy: galaxy.has_mass_profile, self.galaxies)))

    @property
    def galaxies_with_mass_profile(self) -> List[Galaxy]:
        return list(filter(lambda galaxy: galaxy.has_mass_profile, self.galaxies))

    @property
    def mass_profile_list(self) -> List:
        return [
            item
            for mass_profile in self.mass_profile_list_of_galaxies
            for item in mass_profile
        ]

    @property
    def mass_profile_list_of_galaxies(self) -> List:
        return [
            galaxy.mass_profile_list
            for galaxy in self.galaxies
            if galaxy.has_mass_profile
        ]

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        if self.galaxies:
            return sum(
                map(lambda g: g.deflections_yx_2d_from(grid=grid), self.galaxies)
            )
        return np.zeros(shape=(grid.shape[0], 2))

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the convergence of the list of galaxies of the plane's sub-grid, by summing the individual convergences \
        of each galaxy's mass profile.

        The convergence is calculated on the sub-grid and binned-up to the original grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is `True`.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Internally data structures are treated as ndarrays, however the decorator `grid_2d_to_structure` converts
        the output to an `Array2D` using the input `grid`'s attributes.

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
        return np.zeros((grid.shape[0],))

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the potential of the list of galaxies of the plane's sub-grid, by summing the individual potentials \
        of each galaxy's mass profile.

        The potential is calculated on the sub-grid and binned-up to the original grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is `True`.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Internally data structures are treated as ndarrays, however the decorator `grid_2d_to_structure` converts
        the output to an `Array2D` using the input `grid`'s attributes.

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
        return np.zeros((grid.shape[0],))

    @aa.grid_dec.grid_2d_to_structure
    def traced_grid_from(self, grid: aa.type.Grid2DLike) -> aa.type.Grid2DLike:
        """Trace this plane's grid_stacks to the next plane, using its deflection angles."""
        return grid - self.deflections_yx_2d_from(grid=grid)

    def sparse_image_plane_grid_list_from(
        self, grid: aa.type.Grid2DLike, settings_pixelization=aa.SettingsPixelization()
    ) -> Optional[List[aa.type.Grid2DLike]]:

        if not self.has_pixelization:
            return None

        return [
            pixelization.data_pixelization_grid_from(
                data_grid_slim=grid,
                hyper_image=hyper_galaxy_image,
                settings=settings_pixelization,
            )
            for pixelization, hyper_galaxy_image in zip(
                self.pixelization_list, self.hyper_galaxies_with_pixelization_image_list
            )
        ]

    @property
    def has_pixelization(self) -> bool:
        return any([galaxy.pixelization for galaxy in self.galaxies])

    @property
    def galaxies_with_pixelization(self) -> List[Galaxy]:
        return list(filter(lambda galaxy: galaxy.has_pixelization, self.galaxies))

    @property
    def hyper_galaxies_with_pixelization_image_list(self) -> List[Galaxy]:
        return [galaxy.hyper_galaxy_image for galaxy in self.galaxies_with_pixelization]

    @property
    def pixelization_list(self) -> List:
        return [galaxy.pixelization for galaxy in self.galaxies_with_pixelization]

    @property
    def has_regularization(self) -> bool:
        return any([galaxy.regularization for galaxy in self.galaxies])

    @property
    def galaxies_with_regularization(self) -> List[Galaxy]:
        return list(filter(lambda galaxy: galaxy.has_regularization, self.galaxies))

    @property
    def regularization_list(self) -> List:
        return [galaxy.regularization for galaxy in self.galaxies_with_pixelization]

    def mapper_from(
        self,
        source_grid_slim,
        source_pixelization_grid,
        pixelization,
        hyper_galaxy_image,
        data_pixelization_grid=None,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=aa.Preloads(),
    ):

        return pixelization.mapper_from(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_galaxy_image,
            settings=settings_pixelization,
            preloads=preloads,
            profiling_dict=self.profiling_dict,
        )

    def linear_obj_list_from(
        self,
        grid,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=aa.Preloads(),
    ):

        if not self.has_pixelization:
            return None

        sparse_grid_list = self.sparse_image_plane_grid_list_from(grid=grid)

        linear_obj_list = []

        pixelization_list = self.pixelization_list
        hyper_galaxy_image_list = self.hyper_galaxies_with_pixelization_image_list

        for mapper_index in range(len(sparse_grid_list)):

            mapper = self.mapper_from(
                source_grid_slim=grid,
                source_pixelization_grid=sparse_grid_list[mapper_index],
                pixelization=pixelization_list[mapper_index],
                hyper_galaxy_image=hyper_galaxy_image_list[mapper_index],
                data_pixelization_grid=sparse_grid_list[mapper_index],
                settings_pixelization=settings_pixelization,
                preloads=preloads,
            )

            linear_obj_list.append(mapper)

        return linear_obj_list

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

        linear_obj_list = self.linear_obj_list_from(
            grid=grid, settings_pixelization=settings_pixelization, preloads=preloads
        )

        return inversion_imaging_unpacked_from(
            image=image,
            noise_map=noise_map,
            convolver=convolver,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=self.regularization_list,
            settings=settings_inversion,
            preloads=preloads,
            profiling_dict=self.profiling_dict,
        )

    def inversion_interferometer_from(
        self,
        grid,
        visibilities,
        noise_map,
        transformer,
        w_tilde,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion=aa.SettingsInversion(),
        preloads=aa.Preloads(),
    ):

        linear_obj_list = self.linear_obj_list_from(
            grid=grid, settings_pixelization=settings_pixelization, preloads=preloads
        )

        return inversion_interferometer_unpacked_from(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=transformer,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=self.regularization_list,
            settings=settings_inversion,
            preloads=preloads,
            profiling_dict=self.profiling_dict,
        )

    @property
    def has_hyper_galaxy(self) -> bool:
        return any(list(map(lambda galaxy: galaxy.has_hyper_galaxy, self.galaxies)))

    def hyper_noise_map_from(self, noise_map) -> aa.Array2D:
        hyper_noise_maps = self.hyper_noise_map_list_from(noise_map=noise_map)
        return sum(hyper_noise_maps)

    def hyper_noise_map_list_from(self, noise_map) -> List[aa.Array2D]:
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
    def contribution_map(self) -> Optional[aa.Array2D]:

        contribution_map_list = self.contribution_map_list

        contribution_map_list = [i for i in contribution_map_list if i is not None]

        if contribution_map_list:
            return sum(contribution_map_list)
        else:
            return None

    @property
    def contribution_map_list(self) -> List[aa.Array2D]:

        contribution_map_list = []

        for galaxy in self.galaxies:

            if galaxy.hyper_galaxy is not None:

                contribution_map_list.append(galaxy.contribution_map)

            else:

                contribution_map_list.append(None)

        return contribution_map_list

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

    def set_snr_of_snr_light_profiles(
        self,
        grid: aa.type.Grid2DLike,
        exposure_time: float,
        background_sky_level: float = 0.0,
    ):

        for galaxy in self.galaxies:
            for light_profile in galaxy.light_profile_list:
                if isinstance(light_profile, LightProfileSNR):
                    light_profile.set_intensity_from(
                        grid=grid,
                        exposure_time=exposure_time,
                        background_sky_level=background_sky_level,
                    )


class PlaneImage:
    def __init__(self, array, grid):
        self.array = array
        self.grid = grid
