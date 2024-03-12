from autogalaxy.profiles.light.basis import Basis
from autogalaxy.profiles.light.linear.abstract import LightProfileLinear

from autogalaxy.profiles.light import standard as lp


class Sky(Basis):
    def __init__(
        self,
    ):
        """
        The linear sky light profile, representing the background sky emission as a constant sheet of values.
        """
        super().__init__(
            light_profile_list=[SkyPos(), SkyNeg()]
        )



class SkyPos(lp.Sky, LightProfileLinear):
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


class SkyNeg(lp.Sky, LightProfileLinear):
    def __init__(
        self,
    ):
        """
        The linear sky light profile, representing the background sky emission as a constant sheet of values.
        """
        super().__init__(
            intensity=-1.0,
        )

    @property
    def lp_cls(self):
        return lp.Sky