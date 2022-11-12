from autogalaxy.profiles.light_profiles import operated as lp_operated
from autogalaxy.profiles.light_profiles import linear as lp_linear


class EllSersic(lp_linear.EllSersic, lp_operated.LightProfileOperated):

    pass
