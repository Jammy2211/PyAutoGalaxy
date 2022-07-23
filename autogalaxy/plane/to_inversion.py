from typing import Dict, List, Optional, Type, Union

import autoarray as aa

from autoarray.inversion.inversion.factory import inversion_unpacked_from
from autogalaxy.profiles.light_profiles.light_profiles_linear import (
    LightProfileLinearObjFuncList,
)
from autogalaxy.profiles.light_profiles.basis import Basis
from autogalaxy.profiles.light_profiles.light_profiles_linear import LightProfileLinear
from autogalaxy.galaxy.galaxy import Galaxy


class PlaneToInversion:
    def __init__(
        self,
        plane: "Plane",
        dataset: Optional[Union[aa.Imaging, aa.Interferometer]] = None,
        data: Optional[Union[aa.Array2D, aa.Visibilities]] = None,
        noise_map: Optional[Union[aa.Array2D, aa.VisibilitiesNoiseMap]] = None,
        w_tilde: Optional[Union[aa.WTildeImaging, aa.WTildeInterferometer]] = None,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads=aa.Preloads(),
    ):

        self.plane = plane
        self.dataset = dataset
        self.data = data
        self.noise_map = noise_map
        self.w_tilde = w_tilde
        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion
        self.preloads = preloads

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
                            grid=self.dataset.grid,
                            blurring_grid=self.dataset.blurring_grid,
                            convolver=self.dataset.convolver,
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

    def sparse_image_plane_grid_list_from(self,) -> Optional[List[aa.type.Grid2DLike]]:

        if not self.plane.has(cls=aa.pix.Pixelization):
            return None

        return [
            pixelization.data_pixelization_grid_from(
                data_grid_slim=self.dataset.grid_pixelized,
                hyper_image=hyper_galaxy_image,
                settings=self.settings_pixelization,
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
            source_grid_slim=self.dataset.grid_pixelized,
            source_pixelization_grid=source_pixelization_grid,
            data_pixelization_grid=data_pixelization_grid,
            hyper_image=hyper_galaxy_image,
            settings=settings_pixelization,
            preloads=preloads,
            profiling_dict=self.plane.profiling_dict,
        )

    def mapper_galaxy_dict_from(self,) -> Dict[aa.AbstractMapper, Galaxy]:

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
                settings_pixelization=self.settings_pixelization,
                preloads=self.preloads,
            )

            galaxy = galaxies_with_pixelization_list[mapper_index]

            mapper_galaxy_dict[mapper] = galaxy

        return mapper_galaxy_dict

    def linear_obj_galaxy_dict_from(
        self,
    ) -> Dict[Union[LightProfileLinearObjFuncList, aa.AbstractMapper], Galaxy]:

        lp_linear_func_galaxy_dict = self.lp_linear_func_list_galaxy_dict_from()

        mapper_galaxy_dict = self.mapper_galaxy_dict_from()

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

    def inversion_from(self,):

        linear_obj_galaxy_dict = self.linear_obj_galaxy_dict_from()

        linear_obj_list = list(linear_obj_galaxy_dict.keys())

        regularization_list = self.regularization_list_from(
            linear_obj_galaxy_dict=linear_obj_galaxy_dict,
            linear_obj_list=linear_obj_list,
        )

        inversion = inversion_unpacked_from(
            dataset=self.dataset,
            data=self.data,
            noise_map=self.noise_map,
            w_tilde=self.w_tilde,
            linear_obj_list=linear_obj_list,
            regularization_list=regularization_list,
            settings=self.settings_inversion,
            preloads=self.preloads,
            profiling_dict=self.plane.profiling_dict,
        )

        inversion.linear_obj_galaxy_dict = linear_obj_galaxy_dict

        return inversion
