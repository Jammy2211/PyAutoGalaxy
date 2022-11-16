from autogalaxy.profiles.light import operated as lp_operated
from autogalaxy.profiles.light import linear as lp_linear


class EllSersic(lp_linear.EllSersic, lp_operated.LightProfileOperated):

    pass
