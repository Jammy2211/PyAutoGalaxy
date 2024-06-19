import numpy as np
from typing import Dict, List, Optional, Type, Union

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy.profiles.light import linear as lp_linear


class Basis(LightProfile, MassProfile):
    def __init__(
        self,
        light_profile_list: List[Union[LightProfile, MassProfile]],
        regularization: Optional[aa.AbstractRegularization] = None,
    ):
        super().__init__(
            centre=light_profile_list[0].centre,
            ell_comps=light_profile_list[0].ell_comps,
        )

        self.light_profile_list = light_profile_list
        self.regularization = regularization

    @property
    def mass_list(self) -> List:
        """
        Returns a list of objects in the galaxy which are an instance of the input `cls`.

        The optional `cls_filtered` input removes classes of an input instance type.

        For example:

        - If the input is `cls=ag.LightProfile`, a list containing all light profiles in the galaxy is returned.

        - If `cls=ag.LightProfile` and `cls_filtered=ag.LightProfileLinear`, a list of all light profiles
          excluding those which are linear light profiles will be returned.

        Parameters
        ----------
        cls
            The type of class that a list of instances of this class in the galaxy are returned for.
        cls_filtered
            A class type which is filtered and removed from the class list.

        Returns
        -------
            The list of objects in the galaxy that inherit from input `cls`.
        """
        return aa.util.misc.cls_list_from(
            values=self.light_profile_list, cls=MassProfile
        )

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
            for light_profile in self.light_profile_list
        ]

    def convergence_2d_from(
        self, grid: aa.type.Grid2DLike, **kwargs
    ) -> aa.Array2D:

        if len(self.mass_list) > 0:
            return sum([mass.convergence_2d_from(grid=grid) for mass in self.mass_list])
        return np.zeros((grid.shape[0],))

    def potential_2d_from(
        self, grid: aa.type.Grid2DLike, **kwargs
    ) -> aa.Array2D:
        if len(self.mass_list) > 0:
            return sum([mass.potential_2d_from(grid=grid) for mass in self.light_profile_list])
        return np.zeros((grid.shape[0],))


    def deflections_yx_2d_from(
        self, grid: aa.type.Grid2DLike, **kwargs
    ) -> aa.Array2D:
        if len(self.mass_list) > 0:
            return sum([mass.deflections_yx_2d_from(grid=grid) for mass in self.light_profile_list])
        return np.zeros((grid.shape[0], 2))

    def lp_instance_from(self, linear_light_profile_intensity_dict: Dict):
        light_profile_list = []

        for light_profile in self.light_profile_list:
            if isinstance(light_profile, lp_linear.LightProfileLinear):
                light_profile = light_profile.lp_instance_from(
                    linear_light_profile_intensity_dict=linear_light_profile_intensity_dict
                )

            light_profile_list.append(light_profile)

        return Basis(
            light_profile_list=light_profile_list, regularization=self.regularization
        )
