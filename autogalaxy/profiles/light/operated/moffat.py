from autogalaxy.profiles.light import base as lp

from autogalaxy.profiles.light.operated.abstract import LightProfileOperated


class EllMoffat(lp.EllMoffat, LightProfileOperated):

    pass
