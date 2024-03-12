from autogalaxy.profiles.light.linear.abstract import LightProfileLinear

from autogalaxy.profiles.light import standard as lp
from autogalaxy.profiles import light_and_mass_profiles as lmp


class Sky(lp.Sky, LightProfileLinear):
    def __init__(
        self,
    ):
        """
        The linear sky light profile, representing the background sky emission as a constant sheet of values.
        """

        super().__init__(
            intensity=1.0,
        )

    @property
    def lp_cls(self):
        return lp.Sky

    @property
    def lmp_cls(self):
        return lmp.sky
