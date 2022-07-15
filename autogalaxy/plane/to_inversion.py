from typing import Dict, List, Optional, Union

import autoarray as aa

from autoarray.inversion.inversion.factory import inversion_imaging_unpacked_from
from autoarray.inversion.inversion.factory import inversion_interferometer_unpacked_from
from autogalaxy.profiles.light_profiles.light_profiles_linear import (
    LightProfileLinearObjFunc,
)
from autogalaxy.profiles.light_profiles.light_profiles_linear import LightProfileLinear
from autogalaxy.galaxy.galaxy import Galaxy


class PlaneToInversion:
    def __init__(self, plane: "Plane"):

        self.plane = plane

    def lp_linear_func_galaxy_dict_from(
        self,
        source_grid_slim: aa.type.Grid2DLike,
        source_blurring_grid_slim: Optional[aa.type.Grid2DLike],
        convolver: Optional[aa.Convolver] = None,
    ) -> Dict[LightProfileLinearObjFunc, Galaxy]:

        if not self.plane.has(cls=LightProfileLinear):
            return {}

        lp_linear_func_galaxy_dict = {}

        for galaxy in self.plane.galaxies:
            if galaxy.has(cls=LightProfileLinear):
                for light_profile_linear in galaxy.cls_list_from(
                    cls=LightProfileLinear
                ):

                    lp_linear_func = LightProfileLinearObjFunc(
                        grid=source_grid_slim,
                        blurring_grid=source_blurring_grid_slim,
                        convolver=convolver,
                        light_profile=light_profile_linear,
                    )

                    lp_linear_func_galaxy_dict[lp_linear_func] = galaxy

        return lp_linear_func_galaxy_dict

    def sparse_image_plane_grid_list_from(
        self, grid: aa.type.Grid2DLike, settings_pixelization=aa.SettingsPixelization()
    ) -> Optional[List[aa.type.Grid2DLike]]:

        if not self.plane.has(cls=aa.pix.Pixelization):
            return None

        return [
            pixelization.data_pixelization_grid_from(
                data_grid_slim=grid,
                hyper_image=hyper_galaxy_image,
                settings=settings_pixelization,
            )
            for pixelization, hyper_galaxy_image in zip(
                self.plane.cls_list_from(cls=aa.pix.Pixelization),
                self.plane.hyper_galaxies_with_pixelization_image_list,
            )
        ]

    def mapper_from(
        self,
        source_grid_slim: aa.type.Grid2DLike,
        source_pixelization_grid: aa.type.Grid2DLike,
        pixelization: aa.AbstractPixelization,
        hyper_galaxy_image: aa.Array2D,
        data_pixelization_grid: aa.Grid2D = None,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=aa.Preloads(),
    ) -> aa.AbstractMapper:

        return pixelization.mapper_from(
            source_grid_slim=source_grid_slim,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_galaxy_image,
            settings=settings_pixelization,
            preloads=preloads,
            profiling_dict=self.plane.profiling_dict,
        )

    def mapper_galaxy_dict_from(
        self,
        grid: aa.Grid2D,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=aa.Preloads(),
    ) -> Dict[aa.AbstractMapper, Galaxy]:

        if not self.plane.has(cls=aa.pix.Pixelization):
            return {}

        sparse_grid_list = self.sparse_image_plane_grid_list_from(grid=grid)

        mapper_galaxy_dict = {}

        pixelization_list = self.plane.cls_list_from(cls=aa.pix.Pixelization)
        galaxies_with_pixelization_list = self.plane.galaxies_with_cls_list_from(
            cls=aa.pix.Pixelization
        )
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

            galaxy = galaxies_with_pixelization_list[mapper_index]

            mapper_galaxy_dict[mapper] = galaxy

        return mapper_galaxy_dict

    def linear_obj_galaxy_dict_from(
        self,
        dataset: Union[aa.Imaging, aa.Interferometer],
        settings_pixelization=aa.SettingsPixelization(),
        preloads=aa.Preloads(),
    ) -> Dict[Union[LightProfileLinearObjFunc, aa.AbstractMapper], Galaxy]:

        lp_linear_func_galaxy_dict = self.lp_linear_func_galaxy_dict_from(
            source_grid_slim=dataset.grid,
            source_blurring_grid_slim=dataset.blurring_grid,
            convolver=dataset.convolver,
        )

        mapper_galaxy_dict = self.mapper_galaxy_dict_from(
            grid=dataset.grid_pixelized,
            settings_pixelization=settings_pixelization,
            preloads=preloads,
        )

        return {**lp_linear_func_galaxy_dict, **mapper_galaxy_dict}

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

        linear_obj_galaxy_dict = self.linear_obj_galaxy_dict_from(
            dataset=dataset,
            settings_pixelization=settings_pixelization,
            preloads=preloads,
        )

        linear_obj_list = list(linear_obj_galaxy_dict.keys())

        inversion = inversion_imaging_unpacked_from(
            image=image,
            noise_map=noise_map,
            convolver=dataset.convolver,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=self.plane.cls_list_from(cls=aa.reg.Regularization),
            settings=settings_inversion,
            preloads=preloads,
            profiling_dict=self.plane.profiling_dict,
        )

        inversion.linear_obj_galaxy_dict = linear_obj_galaxy_dict

        return inversion

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

        linear_obj_galaxy_dict = self.linear_obj_galaxy_dict_from(
            dataset=dataset,
            settings_pixelization=settings_pixelization,
            preloads=preloads,
        )

        linear_obj_list = list(linear_obj_galaxy_dict.keys())

        inversion = inversion_interferometer_unpacked_from(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=dataset.transformer,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=self.plane.cls_list_from(cls=aa.reg.Regularization),
            settings=settings_inversion,
            preloads=preloads,
            profiling_dict=self.plane.profiling_dict,
        )

        inversion.linear_obj_galaxy_dict = linear_obj_galaxy_dict

        return inversion
