from autogalaxy.profiles.light_profiles import light_profiles as lp
from autogalaxy.profiles.light_profiles import light_profiles_linear as lp_linear

# TODO : Would rather remove `is_operated` and use `isinstance_` when checking in decorator, but currently get import error.


class LightProfileOperated:

    pass


class EllGaussian(lp.EllGaussian, LightProfileOperated):
    @property
    def is_operated(self):
        return True


class EllGaussianLinear(lp_linear.EllGaussian, LightProfileOperated):
    @property
    def is_operated(self):
        return True
