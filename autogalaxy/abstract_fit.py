from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Optional

from autofit import ModelInstance

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy

import autoarray as aa

from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.profiles.light.basis import Basis


class AbstractFitInversion:
    def __init__(self, model_obj, settings_inversion: aa.SettingsInversion):
        """
        An abstract fit object which fits to datasets (e.g. imaging, interferometer) inherit from.

        This object primarily inspects the `model_obj` (e.g. a plane object PyAutoGalaxy or tracer in PyAutoLens)
        and determines the properties used for the fit by inspecting the galaxies / light profiles in this object.

        Parameters
        ----------
        model_obj
            The object which contains the model components (e.g. light profiles, galaxies, etc) which are used to
            create the model-data that fits the data. In PyAutoGalaxy this is a `Plane` and PyAutoLens it is a `Tracer`.
        settings_inversion
            Settings controlling how an inversion is fitted for example which linear algebra formalism is used.
        """
        self.model_obj = model_obj
        self.settings_inversion = settings_inversion

    @property
    def total_mappers(self) -> int:
        """
        The total number of `Mapper` objects used by the inversion in this fit.

        A mapper is created for each galaxy with a pixelization object.
        """
        # TODO : When we add regularization to basis need to change this to reflect mappers.

        return len(
            list(filter(None, self.model_obj.cls_list_from(cls=aa.Pixelization)))
        )

    @property
    def perform_inversion(self) -> bool:
        """
        Returns a bool specifying whether this fit object performs an inversion.

        This is based on whether any of the galaxies in the `model_obj` have a `Pixelization` or `LightProfileLinear`
        object, in which case an inversion is performed.

        Returns
        -------
            A bool which is True if an inversion is performed.
        """

        return self.model_obj.perform_inversion

    @property
    def w_tilde(self) -> Optional[aa.WTildeImaging]:
        """
        Only call the `w_tilde` property of a dataset used to perform efficient linear algebra calcualtions if
        the SettingsInversion()` object has `use_w_tilde=True`, to avoid unnecessary computation.

        Returns
        -------
        The w-tilde matrix if the w-tilde formalism is being used for this inversion.
        """
        if self.settings_inversion.use_w_tilde:
            if self.total_mappers > 0:
                return self.dataset.w_tilde

    @property
    def inversion(self) -> Optional[aa.Inversion]:
        raise NotImplementedError

    @property
    def linear_light_profile_intensity_dict(
        self,
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

        linear_obj_func_list = self.inversion.cls_list_from(
            cls=aa.AbstractLinearObjFuncList
        )

        linear_light_profile_intensity_dict = {}

        for linear_obj_func in linear_obj_func_list:
            reconstruction = self.inversion.reconstruction_dict[linear_obj_func]

            for i, light_profile in enumerate(linear_obj_func.light_profile_list):
                linear_light_profile_intensity_dict[light_profile] = float(
                    reconstruction[i]
                )

        return linear_light_profile_intensity_dict

    def galaxy_linear_obj_data_dict_from(
        self, use_image: bool = False
    ) -> Dict[Galaxy, aa.Array2D]:
        """
        Returns a dictionary mapping every galaxy containing a linear
        object (e.g. a linear light profile / pixelization) in the `model_obj` to the `model_data` of its linear
        objects.

        The `model_data` is the `reconstructed_data` solved for in the inversion`.

        This is used to create the overall `galaxy_model_image_dict`, which maps every galaxy to its
        overall `model_data` (e.g. including the `model_data` of orindary light profiles too).

        If `use_image=False`, the `reconstructed_data` of the inversion (e.g. an image for dataset data,
        visibilities for  interferometer data) is input in the dictionary.

        if `use_image=True`, the `reconstructed_image` of the inversion (e.g. the image for dataset data, the
        real-space image for interferometer data) is input in the dictionary.

        Parameters
        ----------
        use_image
            Whether to put the reconstructed data or images in the dictionary.

        Returns
        -------
        The dictionary mapping all galaxies with linear objects to the model data / images of those linear objects
        reconstructed by the inversion.
        """
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

    @property
    def model_obj_linear_light_profiles_to_light_profiles(self):
        """
        The model object may contain linear light profiles, which solve for the `intensity` during the `Inversion`.

        This means they are difficult to visualize, because they do not have a valid `intensity` parameter.

        To address this, this property creates a new `model_obj` where all linear light profiles are converted to
        ordinary light profiles whose `intensity` parameters are set to the results of the Inversion.

        Returns
        -------
        A `model_obj` (E.g. `Plane` or `Tracer`) where the light profile intensity values are set to the results
        of those inferred via the `Inversion`.
        """

        if self.linear_light_profile_intensity_dict is None:
            return self.model_obj

        model_instance = ModelInstance(dict(model_obj=self.model_obj))

        for path, instance in model_instance.path_instance_tuples_for_class(
            (LightProfileLinear, Basis)
        ):
            model_instance = model_instance.replacing_for_path(
                path,
                instance.lp_instance_from(self.linear_light_profile_intensity_dict),
            )

        return model_instance.model_obj
