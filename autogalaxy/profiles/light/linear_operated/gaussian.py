from autogalaxy.profiles.light import operated as lp_operated
from autogalaxy.profiles.light import linear as lp_linear


class EllGaussian(lp_linear.EllGaussian, lp_operated.LightProfileOperated):

    pass
