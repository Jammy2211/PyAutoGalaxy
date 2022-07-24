from typing import Dict, List, Optional, Type, Union

from autoconf import cached_property

import autoarray as aa

from autoarray.inversion.inversion.factory import inversion_unpacked_from
from autogalaxy.profiles.light_profiles.light_profiles_linear import (
    LightProfileLinearObjFuncList,
)
from autogalaxy.profiles.light_profiles.basis import Basis
from autogalaxy.profiles.light_profiles.light_profiles_linear import LightProfileLinear
from autogalaxy.galaxy.galaxy import Galaxy


class AbstractToInversion:
    def cls_light_profile_func_list_galaxy_dict_from(
        self, cls: Type
    ) -> Dict[LightProfileLinearObjFuncList, Galaxy]:
        raise NotImplementedError

    @cached_property
    def lp_linear_func_list_galaxy_dict(
        self,
    ) -> Dict[LightProfileLinearObjFuncList, Galaxy]:

        raise NotImplementedError

    @cached_property
    def mapper_galaxy_dict(self):
        raise NotImplementedError

    @cached_property
    def linear_obj_galaxy_dict(
        self,
    ) -> Dict[Union[LightProfileLinearObjFuncList, aa.AbstractMapper], Galaxy]:

        lp_linear_func_galaxy_dict = self.lp_linear_func_list_galaxy_dict

        mapper_galaxy_dict = self.mapper_galaxy_dict

        return {**lp_linear_func_galaxy_dict, **mapper_galaxy_dict}

    @cached_property
    def linear_obj_list(self) -> List[aa.LinearObj]:
        return list(self.linear_obj_galaxy_dict.keys())

    @cached_property
    def regularization_list(self,) -> List[aa.reg.Regularization]:

        regularization_list = []

        for linear_obj in self.linear_obj_list:

            regularization = None

            galaxy = self.linear_obj_galaxy_dict[linear_obj]

            if hasattr(linear_obj, "regularization"):
                if linear_obj.regularization is not None:
                    regularization = linear_obj.regularization

            if regularization is None:
                if galaxy.has(cls=aa.reg.Regularization):
                    regularization = galaxy.regularization

            regularization_list.append(regularization)

        return regularization_list


class PlaneToInversion(AbstractToInversion):
    def __init__(
        self,
        plane: "Plane",
        dataset: Optional[Union[aa.Imaging, aa.Interferometer]] = None,
        data: Optional[Union[aa.Array2D, aa.Visibilities]] = None,
        noise_map: Optional[Union[aa.Array2D, aa.VisibilitiesNoiseMap]] = None,
        w_tilde: Optional[Union[aa.WTildeImaging, aa.WTildeInterferometer]] = None,
        grid: Optional[aa.type.Grid2DLike] = None,
        blurring_grid: Optional[aa.type.Grid2DLike] = None,
        grid_pixelized: Optional[aa.type.Grid2DLike] = None,
        settings_pixelization=aa.SettingsPixelization(),
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads=aa.Preloads(),
    ):

        self.plane = plane
        self.dataset = dataset
        self.data = data
        self.noise_map = noise_map
        self.w_tilde = w_tilde

        if grid is not None:
            self.grid = grid
        elif dataset is not None:
            self.grid = dataset.grid
        else:
            self.grid = None

        if blurring_grid is not None:
            self.blurring_grid = blurring_grid
        elif dataset is not None:
            self.blurring_grid = dataset.blurring_grid
        else:
            self.blurring_grid = None

        if grid_pixelized is not None:
            self.grid_pixelized = grid_pixelized
        elif dataset is not None:
            self.grid_pixelized = dataset.grid_pixelized
        else:
            self.grid_pixelized = None

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
                        regularization = None
                    else:
                        light_profile_list = light_profile.light_profile_list
                        light_profile_list = [
                            light_profile
                            for light_profile in light_profile_list
                            if isinstance(light_profile, LightProfileLinear)
                        ]
                        regularization = light_profile.regularization

                    if len(light_profile_list) > 0:

                        lp_linear_func = LightProfileLinearObjFuncList(
                            grid=self.grid,
                            blurring_grid=self.blurring_grid,
                            convolver=self.dataset.convolver,
                            light_profile_list=light_profile_list,
                            regularization=regularization,
                        )

                        lp_linear_func_galaxy_dict[lp_linear_func] = galaxy

        return lp_linear_func_galaxy_dict

    @cached_property
    def lp_linear_func_list_galaxy_dict(
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

    @cached_property
    def sparse_image_plane_grid_list(self,) -> Optional[List[aa.type.Grid2DLike]]:

        if not self.plane.has(cls=aa.Pixelization):
            return None

        return [
            pixelization.mesh.data_pixelization_grid_from(
                data_grid_slim=self.grid_pixelized,
                hyper_image=hyper_galaxy_image,
                settings=self.settings_pixelization,
            )
            for pixelization, hyper_galaxy_image in zip(
                self.plane.cls_list_from(cls=aa.Pixelization),
                self.plane.hyper_galaxies_with_pixelization_image_list,
            )
        ]

    def mapper_from(
        self,
        source_mesh_grid: aa.type.Grid2DLike,
        pixelization: aa.AbstractMesh,
        hyper_galaxy_image: aa.Array2D,
        data_mesh_grid: aa.Grid2D = None,
    ) -> aa.AbstractMapper:

        return pixelization.mapper_from(
            source_grid_slim=self.grid_pixelized,
            source_mesh_grid=source_mesh_grid,
            data_mesh_grid=data_mesh_grid,
            hyper_image=hyper_galaxy_image,
            settings=self.settings_pixelization,
            preloads=self.preloads,
            profiling_dict=self.plane.profiling_dict,
        )

    @cached_property
    def mapper_galaxy_dict(self) -> Dict[aa.AbstractMapper, Galaxy]:

        if not self.plane.has(cls=aa.Pixelization):
            return {}

        sparse_grid_list = self.sparse_image_plane_grid_list

        mapper_galaxy_dict = {}

        pixelization_list = self.plane.cls_list_from(cls=aa.Pixelization)
        galaxies_with_pixelization_list = self.plane.galaxies_with_cls_list_from(
            cls=aa.Pixelization
        )
        hyper_galaxy_image_list = self.plane.hyper_galaxies_with_pixelization_image_list

        for mapper_index in range(len(sparse_grid_list)):

            mapper = self.mapper_from(
                source_mesh_grid=sparse_grid_list[mapper_index],
                pixelization=pixelization_list[mapper_index],
                hyper_galaxy_image=hyper_galaxy_image_list[mapper_index],
                data_mesh_grid=sparse_grid_list[mapper_index],
            )

            galaxy = galaxies_with_pixelization_list[mapper_index]

            mapper_galaxy_dict[mapper] = galaxy

        return mapper_galaxy_dict

    @property
    def inversion(self) -> aa.AbstractInversion:

        inversion = inversion_unpacked_from(
            dataset=self.dataset,
            data=self.data,
            noise_map=self.noise_map,
            w_tilde=self.w_tilde,
            linear_obj_list=self.linear_obj_list,
            regularization_list=self.regularization_list,
            settings=self.settings_inversion,
            preloads=self.preloads,
            profiling_dict=self.plane.profiling_dict,
        )

        inversion.linear_obj_galaxy_dict = self.linear_obj_galaxy_dict

        return inversion
