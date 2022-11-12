from autogalaxy.profiles.light_profiles import operated as lp_operated
from autogalaxy.profiles.light_profiles import linear as lp_linear


class EllMoffat(lp_linear.EllMoffat, lp_operated.LightProfileOperated):

    pass
