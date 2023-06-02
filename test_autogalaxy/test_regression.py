from autogalaxy import Galaxy
from autogalaxy.abstract_fit import AbstractFitInversion
import autofit as af
import autoarray as aa
from autogalaxy.profiles.light.linear import LightProfileLinear, Sersic


class FitInversion(AbstractFitInversion):
    def __init__(
        self, model_obj, settings_inversion: aa.SettingsInversion, light_profiles
    ):
        super().__init__(model_obj, settings_inversion)
        self.light_profiles = light_profiles

    @property
    def linear_light_profile_intensity_dict(self):
        return {light_profile: 1.0 for light_profile in self.light_profiles}


def test_no_modify_state():
    light_profile = Sersic()

    model_obj = af.ModelInstance(
        {
            "galaxies": af.ModelInstance(
                {
                    "galaxy": Galaxy(
                        redshift=0.5,
                        light_profile=light_profile,
                    )
                }
            )
        }
    )

    fit_inversion = FitInversion(
        model_obj=model_obj,
        settings_inversion=aa.SettingsInversion(use_linear_operators=True),
        light_profiles=[light_profile],
    )
    result = fit_inversion.model_obj_linear_light_profiles_to_light_profiles

    assert result.galaxies.galaxy.light_profile is not light_profile
    assert model_obj.galaxies.galaxy.light_profile is light_profile
