from autogalaxy.profiles.light_profiles import light_profiles as lp


class LightProfileOperated:

    pass


class EllGaussian(lp.EllGaussian, LightProfileOperated):

    pass


class EllMoffat(lp.EllMoffat, LightProfileOperated):

    pass


class EllSersic(lp.EllSersic, LightProfileOperated):

    pass
