from autoconf import conf
import autofit as af

from autoarray.inversion import pixelizations as pix, regularization as reg
from autogalaxy.profiles import (
    light_profiles as lp,
    mass_profiles as mp,
    light_and_mass_profiles as lmp,
)
from autogalaxy.hyper import hyper_data as hd
from autogalaxy import exc

from typing import Union, Optional

import copy


class SetupHyper:
    def __init__(
        self,
        hyper_galaxies: bool = False,
        hyper_image_sky: Optional[type(hd.HyperImageSky)] = None,
        hyper_background_noise: Optional[type(hd.HyperBackgroundNoise)] = None,
        search: Optional[af.NonLinearSearch] = None,
        dlogz: Optional[float] = None,
    ):
        """
        The hyper setup of a pipeline, which controls how hyper-features in PyAutoGalaxy template pipelines run,
        for example controlling whether hyper galaxies are used to scale the noise and the non-linear searches used
        in these searchs.

        Users can write their own pipelines which do not use or require the *SetupHyper* class.

        Parameters
        ----------
        hyper_galaxies : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used to scale the
            noise-map of the dataset throughout the fitting.
        hyper_image_sky : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            image's background sky component in the model.
        hyper_background_noise : bool
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            noise-map's background component in the model.
        hyper_search_no_inversion : af.NonLinearSearch or None
            The `NonLinearSearch` used by every inversion search.
        hyper_search_with_inversion : af.NonLinearSearch or None
            The `NonLinearSearch` used by every hyper combined search.
        dlogz : float
            The evidence tolerance of the non-linear searches used in the hyper searchs, whereby higher values will
            lead them to end earlier at the expense of accuracy.
        """

        self.dlogz = dlogz

        if dlogz is not None:
            if search is not None:
                raise exc.PipelineException(
                    "You have manually specified a search in the SetupPipeline, and an dlogz."
                    "You cannot manually specify both - remove one."
                    "(If you want the hyper search to use a specific evidence tolerance, include the evidence"
                    "tolerance in its parameters"
                )

        self.hyper_galaxies = hyper_galaxies

        self.hyper_galaxy_names = None

        if search is None:
            self.search = af.DynestyStatic(
                n_live_points=50, dlogz=self.dlogz, sample="rstagger"
            )
        elif search is not None:
            self.search = search

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

    @property
    def hypers_all_off(self):
        if not self.hyper_galaxies:
            if self.hyper_image_sky is None and self.hyper_background_noise is None:
                return True
        return False

    @property
    def hypers_all_except_image_sky_off(self):
        if not self.hyper_galaxies:
            if self.hyper_background_noise is None:
                return True
        return False

    def hyper_image_sky_from_result(self, result: af.Result, as_model=True):

        if self.hyper_image_sky is not None:
            if as_model:
                if hasattr(result, "hyper"):
                    return result.hyper.model.hyper_image_sky
                return result.model.hyper_image_sky
            if hasattr(result, "hyper"):
                return result.hyper.instance.hyper_image_sky
            return result.instance.hyper_image_sky

    def hyper_background_noise_from_result(self, result: af.Result):

        if self.hyper_background_noise is not None:
            if hasattr(result, "hyper"):
                return result.hyper.instance.hyper_background_noise
            return result.instance.hyper_background_noise


# class SetupSMBH:
#
#     def __init__(
#         self,
#         smbh_prior_model: af.Model(mp.MassProfile) = mp.PointMass,
#         smbh_centre_fixed: bool = True,
#     ):
#         """
#         The setup of a super massive black hole (SMBH) in the mass model of a PyAutoGalaxy template pipeline.
#
#         Users can write their own pipelines which do not use or require the *SetupSMBH* class.
#
#         Parameters
#         ----------
#         smbh_prior_model : af.Model(mp.MassProfile)
#             The `MassProfile` used to model the mass of the super massive black hole.
#         smbh_centre_fixed : bool
#             If True, the super-massive black hole's centre is fixed to a value input by the pipeline, else it is
#             free to vary in the model.
#         """
#         self.smbh_prior_model = self._cls_to_prior_model(cls=smbh_prior_model)
#         self.smbh_centre_fixed = smbh_centre_fixed
#
#     def smbh_from_centre(self, centre, centre_sigma=0.1) -> af.Model:
#         """
#         Returns a `Model` of the `smbh_prior_model` which is fitted for in the mass-model representing a
#         super-massive black-hole (smbh).
#
#         The centre of the smbh is an input parameter of the function, and this centre is either fixed to the input
#         values as an instance or fitted for as a model.
#
#         Parameters
#         ----------
#         centre : (float, float)
#             The centre of the `PointMass` that repreents the super-massive black hole.
#         centre_sigma : float
#             If the centre is free, this is the sigma value of each centre's _GaussianPrior_.
#         """
#
#         smbh = af.Model(mp.PointMass)
#
#         if self.smbh_centre_fixed:
#             smbh.centre = centre
#         else:
#             smbh.centre.centre_0 = af.GaussianPrior(mean=centre[0], sigma=centre_sigma)
#             smbh.centre.centre_1 = af.GaussianPrior(mean=centre[1], sigma=centre_sigma)
#
#         return smbh
