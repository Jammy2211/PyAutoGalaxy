import inspect

from autofit.mapper.prior_model.prior_model import PriorModel

from autoastro import exc
from autoastro.galaxy import Galaxy
from autoastro.profiles import light_profiles
from autoastro.profiles import mass_profiles


def is_light_profile_class(cls):
    """
    Parameters
    ----------
    cls
        Some object

    Returns
    -------
    bool: is_light_profile_class
        True if cls is a class that inherits from light profile
    """
    return inspect.isclass(cls) and issubclass(cls, light_profiles.LightProfile)


def is_mass_profile_class(cls):
    """
    Parameters
    ----------
    cls
        Some object

    Returns
    -------
    bool: is_mass_profile_class
        True if cls is a class that inherits from mass profile
    """
    return inspect.isclass(cls) and issubclass(cls, mass_profiles.MassProfile)


class GalaxyModel(PriorModel):
    """
    @DynamicAttrs
    """

    def __init__(
        self,
        redshift,
        align_centres=False,
        align_axis_ratios=False,
        align_orientations=False,
        pixelization=None,
        regularization=None,
        hyper_galaxy=None,
        **kwargs
    ):
        """Class to produce Galaxy instances from sets of profile classes and other model-fitting attributes (e.g. \
         pixelizations, regularization schemes, hyper_galaxies-galaxyes) using the model mapper.

        Parameters
        ----------
        light_profile_classes: [LightProfile]
            The *LightProfile* classes for which model light profile instances are generated for this galaxy model.
        mass_profile_classes: [MassProfile]
            The *MassProfile* classes for which model mass profile instances are generated for this galaxy model.
        align_centres : bool
            If *True*, the same prior will be used for all the profiles centres, such that all light and / or mass \
            profiles always share the same origin.
        align_axis_ratios : bool
            If *True*, the same prior will be used for all the profiles axis-ratio, such that all light and / or mass \
            profiles always share the same axis-ratio.
        align_orientations : bool
            If *True*, the same prior will be used for all the profiles rotation angles phi, such that all light \
            and / or mass profiles always share the same orientation.
        redshift : float | Type[g.Redshift]
            The redshift of this model galaxy.
        model_redshift : bool
            If *True*, the galaxy redshift will be treated as a free-parameter that is fitted for by the non-linear \
            search.
        pixelization : Pixelization
            The pixelization used to reconstruct the galaxy light and fit the observed if using an inversion.
        regularization : Regularization
            The regularization-scheme used to regularization reconstruct the galaxy light when fitting the observed \
            if using an inversion.
        hyper_galaxy : HyperGalaxy
            A model hyper_galaxies-galaxy used for scaling the observed grid's noise_map.
        """

        super().__init__(
            Galaxy,
            redshift=redshift,
            pixelization=pixelization,
            regularization=regularization,
            hyper_galaxy=hyper_galaxy,
            **kwargs
        )
        profile_models = []

        for name, prior_model in self.prior_model_tuples:
            cls = prior_model.cls
            if is_mass_profile_class(cls) or is_light_profile_class(cls):
                profile_models.append(prior_model)

        if len(profile_models) > 0:
            if align_centres:
                centre = profile_models[0].centre
                for profile_model in profile_models:
                    profile_model.centre = centre

            if align_axis_ratios:
                axis_ratio = profile_models[0].axis_ratio
                for profile_model in profile_models:
                    profile_model.axis_ratio = axis_ratio

            if align_orientations:
                phi = profile_models[0].phi
                for profile_model in profile_models:
                    profile_model.phi = phi

        if pixelization is not None and regularization is None:
            raise exc.PriorException(
                "If the galaxy prior has a pixelization, it must also have a "
                "regularization."
            )
        if pixelization is None and regularization is not None:
            raise exc.PriorException(
                "If the galaxy prior has a regularization, it must also have a "
                "pixelization."
            )
