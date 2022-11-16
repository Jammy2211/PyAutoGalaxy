from autogalaxy.profiles.light import operated as lp_operated
from autogalaxy.profiles.light import linear as lp_linear


class EllMoffat(lp_linear.EllMoffat, lp_operated.LightProfileOperated):

    pass
