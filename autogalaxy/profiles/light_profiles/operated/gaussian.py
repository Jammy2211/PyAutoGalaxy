from autogalaxy.profiles.light_profiles import base as lp

from autogalaxy.profiles.light_profiles.operated.abstract import LightProfileOperated


class EllGaussian(lp.EllGaussian, LightProfileOperated):

    pass
