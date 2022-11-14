from autogalaxy.profiles.light_profiles import base as lp

from autogalaxy.profiles.light_profiles.operated.abstract import LightProfileOperated


class EllMoffat(lp.EllMoffat, LightProfileOperated):

    pass
