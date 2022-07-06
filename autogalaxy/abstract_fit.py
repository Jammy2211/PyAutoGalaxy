from typing import Dict, Optional

from autogalaxy.profiles.light_profiles.light_profiles_linear import LightProfileLinear


class AbstractFit:
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