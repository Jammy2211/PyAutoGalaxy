from autogalaxy.profiles.light_profiles import light_profiles_operated as lp_operated
from autogalaxy.profiles.light_profiles import light_profiles_linear as lp_linear


class EllSersic(lp_linear.EllSersic, lp_operated.LightProfileOperated):

    pass