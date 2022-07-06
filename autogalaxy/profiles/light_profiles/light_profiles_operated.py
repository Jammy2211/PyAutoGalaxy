from autogalaxy.profiles.light_profiles import light_profiles as lp

# TODO : Would rather remove `is_operated` and use `isinstance_` when checking in decorator, but currently get import error.


class LightProfileOperated:

    pass


class EllGaussian(lp.EllGaussian, LightProfileOperated):
    @property
    def is_operated(self):
        return True


class EllSersic(lp.EllSersic, LightProfileOperated):
    @property
    def is_operated(self):
        return True
