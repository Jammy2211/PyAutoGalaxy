import inspect
import numpy as np
from typing import Dict, List, Optional

from autoconf import cached_property
import autoarray as aa
import autofit as af

from autogalaxy.profiles.light.operated.abstract import (
    LightProfileOperated,
)

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles import light_and_mass_profiles as lmp

from autogalaxy import exc


class LightProfileLinear(LightProfile):
    @property
    def regularization(self):
        return None

    @property
    def _intensity(self):
        return 1.0

    @property
    def lp_cls(self):
        raise NotImplementedError

    @property
    def lmp_cls(self):
        raise NotImplementedError

    def parameters_dict_from(self, intensity: float) -> Dict[str, float]:
        """
        Returns a dictionary of the parameters of the linear light profile with the `intensity` added.

        This `intenisty` will likely have come from the value inferred via the linear inversion.

        Parameters
        ----------
        intensity
            Overall intensity normalisation of the not linear light profile that is created (units are dimensionless
            and derived from the data the light profile's image is compared too, which is expected to be electrons
            per second).
        """
        parameters_dict = vars(self)

        args = inspect.getfullargspec(self.lp_cls.__init__).args
        args.remove("self")

        parameters_dict = {key: parameters_dict[key] for key in args}
        parameters_dict["intensity"] = intensity

        return parameters_dict

    def lp_instance_from(
        self, linear_light_profile_intensity_dict: Dict
    ) -> LightProfile:
        """
        Creates an instance of a linear light profile using its parent normal light profile (e.g. the non linear
        variant which has an `intensity` parameter).

        The `intensity` value of the profile created is passed into this function and used.

        Parameters
        ----------
        intensity
            Overall intensity normalisation of the not linear light profile that is created (units are dimensionless
            and derived from the data the light profile's image is compared too, which is expected to be electrons
            per second).
        """
        intensity = linear_light_profile_intensity_dict[self]
        parameters_dict = self.parameters_dict_from(intensity=intensity)

        return self.lp_cls(**parameters_dict)

    def lmp_model_from(
        self, linear_light_profile_intensity_dict: Dict
    ) -> af.Model(lmp.LightMassProfile):
        """
        Creates an instance of a linear light profile using its parent light and mass profile (e.g. the non linear
        variant which has `mass_to_light_ratio` and `intensity` parameters).

        The `intensity` value of the profile created is passed into this function and used.

        Parameters
        ----------
        intensity
            Overall intensity normalisation of the not linear light profile that is created (units are dimensionless
            and derived from the data the light profile's image is compared too, which is expected to be electrons
            per second).
        """
        intensity = linear_light_profile_intensity_dict[self]
        parameters_dict = self.parameters_dict_from(intensity=intensity)

        return af.Model(self.lmp_cls, **parameters_dict)


class LightProfileLinearObjFuncList(aa.AbstractLinearObjFuncList):
    def __init__(
        self,
        grid: aa.type.Grid1D2DLike,
        blurring_grid: aa.type.Grid1D2DLike,
        convolver: Optional[aa.Convolver],
        light_profile_list: List[LightProfileLinear],
        regularization=aa.reg.Regularization,
        run_time_dict: Optional[Dict] = None,
    ):
        for light_profile in light_profile_list:
            if not isinstance(light_profile, LightProfileLinear):
                raise exc.ProfileException(
                    """
                    A light profile that is not a LightProfileLinear object has been input into the
                    LightProfileLinearObjFuncList object.

                    Only children of the LightProfileLinear class can be used in a linear inversion.
                    """
                )

        super().__init__(
            grid=grid, regularization=regularization, run_time_dict=run_time_dict
        )

        self.blurring_grid = blurring_grid
        self.convolver = convolver
        self.light_profile_list = light_profile_list

    @property
    def params(self):
        return len(self.light_profile_list)

    @property
    def mapping_matrix(self) -> np.ndarray:
        mapping_matrix = np.zeros(shape=(self.grid.mask.pixels_in_mask, self.params))

        for pixel, light_profile in enumerate(self.light_profile_list):
            image_2d = light_profile.image_2d_from(grid=self.grid).binned.slim

            mapping_matrix[:, pixel] = image_2d

        return mapping_matrix

    @cached_property
    def operated_mapping_matrix_override(self) -> Optional[np.ndarray]:
        """
        The `LinearEqn` object takes the `mapping_matrix` of each linear object and combines it with the `Convolver`
        operator to perform a 2D convolution and compute the `operated_mapping_matrix`.

        If this property is overwritten this operation is not performed, with the `operated_mapping_matrix` output this
        property automatically used instead.

        This is used for a linear light profile because the in-built mapping matrix convolution does not account for
        how light profile images have flux outside the masked region which is blurred into the masked region. This
        flux is outside the region that defines the `mapping_matrix` and thus this override is required to properly
        incorporate it.

        Returns
        -------
        A blurred mapping matrix of dimensions (total_mask_pixels, 1) which overrides the mapping matrix calculations
        performed in the linear equation solvers.
        """

        if isinstance(self.light_profile_list[0], LightProfileOperated):
            return self.mapping_matrix

        operated_mapping_matrix = np.zeros(
            shape=(self.grid.mask.pixels_in_mask, self.params)
        )

        for pixel, light_profile in enumerate(self.light_profile_list):
            image_2d = light_profile.image_2d_from(grid=self.grid)

            blurring_image_2d = light_profile.image_2d_from(grid=self.blurring_grid)

            blurred_image_2d = self.convolver.convolve_image(
                image=image_2d, blurring_image=blurring_image_2d
            )

            operated_mapping_matrix[:, pixel] = blurred_image_2d

        return operated_mapping_matrix
