from typing import Dict, List, Optional, Type, Union

import autoarray as aa

from autoarray.inversion.inversion.factory import inversion_imaging_unpacked_from
from autoarray.inversion.inversion.factory import inversion_interferometer_unpacked_from
from autogalaxy.profiles.light_profiles.light_profiles_linear import (
    LightProfileLinearObjFuncList,
)
from autogalaxy.profiles.light_profiles.basis import Basis
from autogalaxy.profiles.light_profiles.light_profiles_linear import LightProfileLinear
from autogalaxy.galaxy.galaxy import Galaxy

from autogalaxy import exc


class PlaneToInversion:
    def __init__(
        self,
        plane: "Plane",
        grid: Optional[aa.type.Grid2DLike] = None,
        blurring_grid: Optional[aa.type.Grid2DLike] = None,
        convolver: Optional[aa.Convolver] = None,
        grid_pixelized: Optional[aa.type.Grid2DLike] = None,
    ):

        self.plane = plane
        self.grid = grid
        self.blurring_grid = blurring_grid
        self.convolver = convolver
        self.grid_pixelized = grid_pixelized

    def cls_light_profile_func_list_galaxy_dict_from(
        self, cls: Type
    ) -> Dict[LightProfileLinearObjFuncList, Galaxy]:

        if not self.plane.has(cls=cls):
            return {}

        lp_linear_func_galaxy_dict = {}

        for galaxy in self.plane.galaxies:
            if galaxy.has(cls=cls):
                for light_profile in galaxy.cls_list_from(cls=cls):

                    if isinstance(light_profile, LightProfileLinear):
                        light_profile_list = [light_profile]
                    else:
                        light_profile_list = light_profile.light_profile_list
                        light_profile_list = [
                            light_profile
                            for light_profile in light_profile_list
                            if isinstance(light_profile, LightProfileLinear)
                        ]

                    if len(light_profile_list) > 0:

                        lp_linear_func = LightProfileLinearObjFuncList(
                            grid=self.grid,
                            blurring_grid=self.blurring_grid,
                            convolver=self.convolver,
                            light_profile_list=light_profile_list,
                        )

                        lp_linear_func_galaxy_dict[lp_linear_func] = galaxy

        return lp_linear_func_galaxy_dict

    def lp_linear_func_list_galaxy_dict_from(
        self,
    ) -> Dict[LightProfileLinearObjFuncList, Galaxy]:

        lp_linear_light_profile_func_list_galaxy_dict = self.cls_light_profile_func_list_galaxy_dict_from(
            cls=LightProfileLinear
        )

        lp_basis_func_list_galaxy_dict = self.cls_light_profile_func_list_galaxy_dict_from(
            cls=Basis
        )

        return {
            **lp_linear_light_profile_func_list_galaxy_dict,
            **lp_basis_func_list_galaxy_dict,
        }

    def sparse_image_plane_grid_list_from(
        self, settings_pixelization=aa.SettingsPixelization()
    ) -> Optional[List[aa.type.Grid2DLike]]:

        if not self.plane.has(cls=aa.pix.Pixelization):
            return None

        return [
            pixelization.data_pixelization_grid_from(
                data_grid_slim=self.grid_pixelized,
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
        source_pixelization_grid: aa.type.Grid2DLike,
        pixelization: aa.AbstractPixelization,
        hyper_galaxy_image: aa.Array2D,
        data_pixelization_grid: aa.Grid2D = None,
        settings_pixelization=aa.SettingsPixelization(),
        preloads=aa.Preloads(),
    ) -> aa.AbstractMapper:

        return pixelization.mapper_from(
            source_grid_slim=self.grid_pixelized,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_galaxy_image,
            settings=settings_pixelization,
            preloads=preloads,
            profiling_dict=self.plane.profiling_dict,
        )

    def mapper_galaxy_dict_from(
        self, settings_pixelization=aa.SettingsPixelization(), preloads=aa.Preloads()
    ) -> Dict[aa.AbstractMapper, Galaxy]:

        if not self.plane.has(cls=aa.pix.Pixelization):
            return {}

        sparse_grid_list = self.sparse_image_plane_grid_list_from()

        mapper_galaxy_dict = {}

        pixelization_list = self.plane.cls_list_from(cls=aa.pix.Pixelization)
        galaxies_with_pixelization_list = self.plane.galaxies_with_cls_list_from(
            cls=aa.pix.Pixelization
        )
        hyper_galaxy_image_list = self.plane.hyper_galaxies_with_pixelization_image_list

        for mapper_index in range(len(sparse_grid_list)):

            mapper = self.mapper_from(
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
    ) -> Dict[Union[LightProfileLinearObjFuncList, aa.AbstractMapper], Galaxy]:

        lp_linear_func_galaxy_dict = self.lp_linear_func_list_galaxy_dict_from()

        mapper_galaxy_dict = self.mapper_galaxy_dict_from(
            settings_pixelization=settings_pixelization, preloads=preloads
        )

        return {**lp_linear_func_galaxy_dict, **mapper_galaxy_dict}

    def regularization_list_from(
        self, linear_obj_galaxy_dict, linear_obj_list
    ) -> List[aa.reg.Regularization]:

        regularization_list = []

        for linear_obj in linear_obj_list:

            galaxy = linear_obj_galaxy_dict[linear_obj]

            if galaxy.has(cls=aa.reg.Regularization):
                regularization_list.append(galaxy.regularization)
            else:
                regularization_list.append(None)

        return regularization_list

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

        regularization_list = self.regularization_list_from(
            linear_obj_galaxy_dict=linear_obj_galaxy_dict,
            linear_obj_list=linear_obj_list,
        )

        inversion = inversion_imaging_unpacked_from(
            image=image,
            noise_map=noise_map,
            convolver=dataset.convolver,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=regularization_list,
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

        regularization_list = self.regularization_list_from(
            linear_obj_galaxy_dict=linear_obj_galaxy_dict,
            linear_obj_list=linear_obj_list,
        )

        inversion = inversion_interferometer_unpacked_from(
            visibilities=visibilities,
            noise_map=noise_map,
            transformer=dataset.transformer,
            w_tilde=w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=regularization_list,
            settings=settings_inversion,
            preloads=preloads,
            profiling_dict=self.plane.profiling_dict,
        )

        inversion.linear_obj_galaxy_dict = linear_obj_galaxy_dict

        return inversion
