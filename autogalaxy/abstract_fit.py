from typing import Dict, Optional

import autoarray as aa

from autogalaxy.profiles.light_profiles.light_profiles_linear import LightProfileLinear


class AbstractFit:

    def __init__(self, model_obj, settings_inversion):

        self.model_obj = model_obj
        self.settings_inversion = settings_inversion

    @property
    def total_mappers(self):

        # TODO : When we add regularization to basis need to change this to reflect mappers.

        return len(
            list(filter(None, self.model_obj.cls_list_from(cls=aa.reg.Regularization)))
        )

    @property
    def perform_inversion(self) -> bool:
        """
        Returns a bool specifying whether this fit object performs an inversion.

        This is based on whether any of the galaxies in the tracer have a `Pixelization` or `LightProfileLinear`
        object, in which case an inversion is performed.

        Returns
        -------
            A bool which is True if an inversion is performed.
        """
        if self.model_obj.has(cls=aa.pix.Pixelization) or self.model_obj.has(
                cls=LightProfileLinear
        ):
            return True
        return False

    @property
    def w_tilde(self) -> Optional[aa.WTildeImaging]:
        """
        Only call the `w_tilde` property of a dataset if the SettingsInversion()` object has `use_w_tilde=True`,
        to avoid unecessary computation.

        Returns
        -------
            The w-tilde matrix if the w-tilde formalism is being used for this inversion.
        """
        if self.settings_inversion.use_w_tilde:
            if self.total_mappers > 0:
                return self.dataset.w_tilde

    @property
    def inversion(self):
        raise NotImplementedError

    @property
    def linear_light_profile_intensity_dict(
        self
    ) -> Optional[Dict[LightProfileLinear, float]]:
        """
        When linear light profiles are used in an inversion, their `intensity` parameter values are solved for via
        linear algebra.

        These values are contained in the `reconstruction` ndarray of the inversion, however their location in this
        ndarray depends how the inversion was performed.

        This function returns a dictionary which maps every linear light profile instance to its solved for
        `intensity` value in the inversion, so that the intensity value of every light profile can be accessed.
        """
        if self.inversion is None:
            return None

        linear_obj_func_list = self.inversion.linear_obj_func_list

        linear_light_profile_intensity_dict = {}

        for linear_obj_func in linear_obj_func_list:

            linear_light_profile_intensity_dict[linear_obj_func.light_profile] = float(
                self.inversion.reconstruction_dict[linear_obj_func]
            )

        return linear_light_profile_intensity_dict

    def galaxy_linear_obj_data_dict_from(self, use_image: bool = False):

        if self.inversion is None:
            return {}

        galaxy_linear_obj_image_dict = {}

        for linear_obj in self.inversion.linear_obj_list:

            galaxy = self.inversion.linear_obj_galaxy_dict[linear_obj]

            if not use_image:

                mapped_reconstructed = self.inversion.mapped_reconstructed_data_dict[
                    linear_obj
                ]

            else:

                mapped_reconstructed = self.inversion.mapped_reconstructed_image_dict[
                    linear_obj
                ]

            if galaxy in galaxy_linear_obj_image_dict:

                galaxy_linear_obj_image_dict[galaxy] += mapped_reconstructed

            else:

                galaxy_linear_obj_image_dict.update({galaxy: mapped_reconstructed})

        return galaxy_linear_obj_image_dict
