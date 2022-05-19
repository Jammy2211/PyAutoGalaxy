from typing import List, Optional

import autoarray as aa

from autoarray.inversion.inversion.factory import inversion_imaging_unpacked_from
from autoarray.inversion.inversion.factory import inversion_interferometer_unpacked_from
from autogalaxy.profiles.light_profiles.light_profiles_linear import (
    LightProfileLinearObjFunc,
)


class PlaneToInversion:
    def __init__(self, plane):

        self.plane = plane

    def sparse_image_plane_grid_list_from(
        self, grid: aa.type.Grid2DLike, settings_pixelization=aa.SettingsPixelization()
    ) -> Optional[List[aa.type.Grid2DLike]]:

        if not self.plane.has_pixelization:
            return None

        return [
            pixelization.data_pixelization_grid_from(
                data_grid_slim=grid,
                hyper_image=hyper_galaxy_image,
                settings=settings_pixelization,
            )
            for pixelization, hyper_galaxy_image in zip(
                self.plane.pixelization_list,
                self.plane.hyper_galaxies_with_pixelization_image_list,
            )
        ]

    def light_profile_linear_func_list_from(
        self,
        source_grid_slim,
        source_blurring_grid_slim,
        convolver: Optional[aa.Convolver] = None,
    ):

        if not self.plane.has_light_profile_linear:
            return []

        light_profile_linear_list = []

        for galaxy in self.plane.galaxies:
            if galaxy.has_light_profile_linear:
                for light_profile_linear in galaxy.light_profile_linear_list:

                    light_profile_linear_func = LightProfileLinearObjFunc(
                        grid=source_grid_slim,
                        blurring_grid=source_blurring_grid_slim,
                        convolver=convolver,
                        light_profile=light_profile_linear,
                    )

                    light_profile_linear_list.append(light_profile_linear_func)

        return light_profile_linear_list

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
            profiling_dict=self.plane.profiling_dict,
        )

    def mapper_list_from(
        self,
        grid,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=aa.Preloads(),
    ):

        if not self.plane.has_pixelization:
            return []

        sparse_grid_list = self.sparse_image_plane_grid_list_from(grid=grid)

        mapper_list = []

        pixelization_list = self.plane.pixelization_list
        hyper_galaxy_image_list = self.plane.hyper_galaxies_with_pixelization_image_list

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

            mapper_list.append(mapper)

        return mapper_list

    def inversion_imaging_from(
        self,
        dataset: aa.Imaging,
        image: aa.Array2D,
        noise_map: aa.Array2D,
        w_tilde: aa.WTildeImaging,
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads: aa.Preloads = aa.Preloads(),
    ):

        mapper_list = self.mapper_list_from(
            grid=dataset.grid_inversion,
            settings_pixelization=settings_pixelization,
            preloads=preloads,
        )

        light_profile_linear_func_list = self.light_profile_linear_func_list_from(
            source_grid_slim=dataset.grid,
            source_blurring_grid_slim=dataset.blurring_grid,
            convolver=dataset.convolver,
        )

        linear_obj_list = mapper_list + light_profile_linear_func_list

        if self.plane.has_light_profile_linear and settings_inversion.use_w_tilde:
            raise aa.exc.InversionException(
                "Cannot use linear light profiles with w_tilde on."
            )

        return inversion_imaging_unpacked_from(
            image=image,
            noise_map=noise_map,
            convolver=dataset.convolver,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=self.plane.regularization_list,
            settings=settings_inversion,
            preloads=preloads,
            profiling_dict=self.plane.profiling_dict,
        )

    def inversion_interferometer_from(
        self,
        dataset: aa.Interferometer,
        visibilities: aa.Visibilities,
        noise_map: aa.VisibilitiesNoiseMap,
        w_tilde,
        settings_pixelization: aa.SettingsPixelization = aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads: aa.Preloads = aa.Preloads(),
    ):

        mapper_list = self.mapper_list_from(
            grid=dataset.grid_inversion,
            settings_pixelization=settings_pixelization,
            preloads=preloads,
        )

        light_profile_linear_func_list = self.light_profile_linear_func_list_from(
            source_grid_slim=dataset.grid, source_blurring_grid_slim=None
        )

        linear_obj_list = mapper_list + light_profile_linear_func_list

        if self.plane.has_light_profile_linear and settings_inversion.use_w_tilde:
            raise aa.exc.InversionException(
                "Cannot use linear light profiles with w_tilde on."
            )

        return inversion_interferometer_unpacked_from(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=dataset.transformer,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=self.plane.regularization_list,
            settings=settings_inversion,
            preloads=preloads,
            profiling_dict=self.plane.profiling_dict,
        )
