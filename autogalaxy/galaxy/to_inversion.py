from __future__ import annotations
from typing import Dict, List, Optional, Type, Union

from autoconf import cached_property

import autoarray as aa

from autoarray.inversion.pixelization.mappers.factory import mapper_from
from autoarray.inversion.inversion.factory import inversion_from
from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.profiles.light.linear import (
    LightProfileLinearObjFuncList,
)
from autogalaxy.profiles.light.basis import Basis
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.analysis.preloads import Preloads


class AbstractToInversion:
    def __init__(
        self,
        dataset: Optional[Union[aa.Imaging, aa.Interferometer, aa.DatasetInterface]],
        adapt_images: Optional[AdaptImages] = None,
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads=Preloads(),
        run_time_dict: Optional[Dict] = None,
    ):
        if dataset is not None:
            if dataset.noise_covariance_matrix is not None:
                raise aa.exc.InversionException(
                    """
                    You cannot perform an inversion (e.g. use a linear light profile or pixelization) 
                    if the dataset has a `noise_covariance_matrix`.
                    
                    This is because the linear algebra implementation is only valid under the assumption 
                    of independent gaussian noise.
                    """
                )

        self.dataset = dataset

        self.adapt_images = adapt_images

        self.settings_inversion = settings_inversion

        self.preloads = preloads
        self.run_time_dict = run_time_dict

    @property
    def convolver(self):
        try:
            return self.dataset.convolver
        except AttributeError:
            return None

    @property
    def transformer(self):
        try:
            return self.dataset.transformer
        except AttributeError:
            return None

    @property
    def border_relocator(self):
        if self.settings_inversion.use_border_relocator:
            return self.dataset.border_relocator

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
    def regularization_list(self) -> List[aa.AbstractRegularization]:
        return [linear_obj.regularization for linear_obj in self.linear_obj_list]


class GalaxiesToInversion(AbstractToInversion):
    def __init__(
        self,
        dataset: Optional[Union[aa.Imaging, aa.Interferometer, aa.DatasetInterface]],
        galaxies: List[Galaxy],
        adapt_images: Optional[AdaptImages] = None,
        settings_inversion: aa.SettingsInversion = aa.SettingsInversion(),
        preloads=aa.Preloads(),
        run_time_dict: Optional[Dict] = None,
    ):
        self.galaxies = Galaxies(galaxies)

        super().__init__(
            dataset=dataset,
            adapt_images=adapt_images,
            settings_inversion=settings_inversion,
            preloads=preloads,
            run_time_dict=run_time_dict,
        )

    @property
    def has_mapper(self):
        if self.galaxies.has(cls=aa.Pixelization):
            return True

    def cls_light_profile_func_list_galaxy_dict_from(
        self, cls: Type
    ) -> Dict[LightProfileLinearObjFuncList, Galaxy]:
        if not self.galaxies.has(cls=cls):
            return {}

        lp_linear_func_galaxy_dict = {}

        for galaxy in self.galaxies:
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
                            regularization=light_profile.regularization,
                        )

                        lp_linear_func_galaxy_dict[lp_linear_func] = galaxy

        return lp_linear_func_galaxy_dict

    @cached_property
    def lp_linear_func_list_galaxy_dict(
        self,
    ) -> Dict[LightProfileLinearObjFuncList, Galaxy]:
        lp_linear_light_profile_func_list_galaxy_dict = (
            self.cls_light_profile_func_list_galaxy_dict_from(cls=LightProfileLinear)
        )

        lp_basis_func_list_galaxy_dict = (
            self.cls_light_profile_func_list_galaxy_dict_from(cls=Basis)
        )

        return {
            **lp_linear_light_profile_func_list_galaxy_dict,
            **lp_basis_func_list_galaxy_dict,
        }

    @cached_property
    def image_plane_mesh_grid_list(
        self,
    ) -> Optional[List[aa.Grid2DIrregular]]:
        if not self.galaxies.galaxy_has_cls(cls=aa.Pixelization):
            return None

        image_plane_mesh_grid_list = []

        for galaxy in self.galaxies.galaxies_with_cls_list_from(cls=aa.Pixelization):
            pixelization = galaxy.cls_list_from(cls=aa.Pixelization)[0]

            if pixelization.image_mesh is not None:
                try:
                    adapt_data = self.adapt_images.galaxy_image_dict[galaxy]
                except (AttributeError, KeyError):
                    adapt_data = None

                    if pixelization.image_mesh.uses_adapt_images:
                        raise aa.exc.PixelizationException(
                            """
                            Attempted to perform fit using a pixelization which requires an 
                            image-mesh (E.g. KMeans, Hilbert).
                            
                            However, the adapt-images passed to the fit (E.g. FitImaging, FitInterferometer) 
                            is None. Without an adapt image, an image-mesh cannot be used.
                            """
                        )

                image_plane_mesh_grid = (
                    pixelization.image_mesh.image_plane_mesh_grid_from(
                        mask=self.dataset.mask,
                        adapt_data=adapt_data,
                        settings=self.settings_inversion,
                    )
                )

            else:
                image_plane_mesh_grid = None

            image_plane_mesh_grid_list.append(image_plane_mesh_grid)

        return image_plane_mesh_grid_list

    def mapper_from(
        self,
        mesh: aa.AbstractMesh,
        regularization: aa.AbstractRegularization,
        source_plane_mesh_grid: aa.Grid2DIrregular,
        source_plane_data_grid: aa.Grid2D,
        adapt_galaxy_image: aa.Array2D,
        image_plane_mesh_grid: Optional[aa.Grid2DIrregular] = None,
    ) -> aa.AbstractMapper:
        mapper_grids = mesh.mapper_grids_from(
            mask=self.dataset.mask,
            border_relocator=self.border_relocator,
            source_plane_data_grid=source_plane_data_grid,
            source_plane_mesh_grid=source_plane_mesh_grid,
            image_plane_mesh_grid=image_plane_mesh_grid,
            adapt_data=adapt_galaxy_image,
            preloads=self.preloads,
            run_time_dict=self.run_time_dict,
        )

        return mapper_from(
            mapper_grids=mapper_grids,
            over_sampler=self.dataset.grid_pixelization.over_sampling.over_sampler_from(
                mask=self.dataset.mask
            ),
            regularization=regularization,
            run_time_dict=self.run_time_dict,
        )

    @cached_property
    def mapper_galaxy_dict(self) -> Dict[aa.AbstractMapper, Galaxy]:
        if not self.galaxies.galaxy_has_cls(cls=aa.Pixelization):
            return {}

        mesh_grid_list = self.image_plane_mesh_grid_list

        mapper_galaxy_dict = {}

        pixelization_list = []

        galaxies_with_pixelization_list = self.galaxies.galaxies_with_cls_list_from(
            cls=aa.Pixelization
        )

        for pix in self.galaxies.cls_list_from(cls=aa.Pixelization):
            pixelization_list.append(pix)

        for mapper_index in range(len(mesh_grid_list)):
            galaxy = galaxies_with_pixelization_list[mapper_index]

            try:
                adapt_galaxy_image = self.adapt_images.galaxy_image_dict[galaxy]
            except (AttributeError, KeyError):
                adapt_galaxy_image = None

            mapper = self.mapper_from(
                mesh=pixelization_list[mapper_index].mesh,
                regularization=pixelization_list[mapper_index].regularization,
                source_plane_data_grid=self.dataset.grid_pixelization.over_sampler.over_sampled_grid,
                source_plane_mesh_grid=mesh_grid_list[mapper_index],
                adapt_galaxy_image=adapt_galaxy_image,
                image_plane_mesh_grid=mesh_grid_list[mapper_index],
            )

            mapper_galaxy_dict[mapper] = galaxy

        return mapper_galaxy_dict

    @property
    def inversion(self) -> aa.AbstractInversion:
        inversion = inversion_from(
            dataset=self.dataset,
            linear_obj_list=self.linear_obj_list,
            settings=self.settings_inversion,
            preloads=self.preloads,
            run_time_dict=self.run_time_dict,
        )

        inversion.linear_obj_galaxy_dict = self.linear_obj_galaxy_dict

        return inversion
