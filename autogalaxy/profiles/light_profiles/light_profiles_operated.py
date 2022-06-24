from autogalaxy.profiles.light_profiles import light_profiles as lp
from autogalaxy.profiles.light_profiles import light_profiles_linear as lp_linear


class LightProfileOperated:

    pass


class EllGaussian(lp.EllGaussian, LightProfileOperated):

    pass


class EllGaussianLinear(lp_linear.EllGaussian, LightProfileOperated):

    pass
