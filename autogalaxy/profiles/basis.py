import numpy as np
from typing import Dict, List, Optional, Type, Union

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy.profiles.light import linear as lp_linear


class Basis(LightProfile, MassProfile):
    def __init__(
        self,
        profile_list: List[Union[LightProfile, MassProfile]],
        regularization: Optional[aa.AbstractRegularization] = None,
    ):
        super().__init__(
            centre=profile_list[0].centre,
            ell_comps=profile_list[0].ell_comps,
        )

        self.profile_list = profile_list
        self.regularization = regularization

    @property
    def light_profile_list(self) -> List[LightProfile]:
        """
        Returns a list of all light profiles in the `Basis` object.

        This is used for computing light profile quantities of each individual light profile in the `Basis` object and
        then summing them to get the overall quantity (e.g. the image, surface brightness, etc.).

        Returns
        -------
            The list of light profiles in the `Basis` object.
        """
        return aa.util.misc.cls_list_from(values=self.profile_list, cls=LightProfile)

    @property
    def mass_profile_list(self) -> List[MassProfile]:
        """
        Returns a list of all mass profiles in the `Basis` object.

        This is used for computing mass profile quantities of each individual mass profile in the `Basis` object and
        then summing them to get the overall quantity (e.g. the convergence, potential, etc.).

        Returns
        -------
            The list of mass profiles in the `Basis` object.
        """
        return aa.util.misc.cls_list_from(values=self.profile_list, cls=MassProfile)

    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None, **kwargs
    ) -> aa.Array2D:
        return sum(self.image_2d_list_from(grid=grid, operated_only=operated_only))

    def image_2d_list_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> List[aa.Array2D]:
        return [
            light_profile.image_2d_from(grid=grid, operated_only=operated_only)
            if not isinstance(light_profile, lp_linear.LightProfileLinear)
            else np.zeros((grid.shape[0],))
            for light_profile in self.profile_list
        ]

    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs) -> aa.Array2D:
        if len(self.mass_profile_list) > 0:
            return sum(
                [mass.convergence_2d_from(grid=grid) for mass in self.mass_profile_list]
            )
        return np.zeros((grid.shape[0],))

    def potential_2d_from(self, grid: aa.type.Grid2DLike, **kwargs) -> aa.Array2D:
        if len(self.mass_profile_list) > 0:
            return sum(
                [mass.potential_2d_from(grid=grid) for mass in self.profile_list]
            )
        return np.zeros((grid.shape[0],))

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs) -> aa.Array2D:
        if len(self.mass_profile_list) > 0:
            return sum(
                [mass.deflections_yx_2d_from(grid=grid) for mass in self.profile_list]
            )
        return np.zeros((grid.shape[0], 2))

    def lp_instance_from(self, linear_light_profile_intensity_dict: Dict):
        light_profile_list = []

        for light_profile in self.profile_list:
            if isinstance(light_profile, lp_linear.LightProfileLinear):
                light_profile = light_profile.lp_instance_from(
                    linear_light_profile_intensity_dict=linear_light_profile_intensity_dict
                )

            light_profile_list.append(light_profile)

        return Basis(
            profile_list=light_profile_list, regularization=self.regularization
        )
