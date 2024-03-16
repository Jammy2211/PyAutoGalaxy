from autogalaxy.profiles.light.basis import Basis
from autogalaxy.profiles.light.linear.abstract import LightProfileLinear

from autogalaxy.profiles.light import standard as lp


class Sky(Basis):
    def __init__(
        self,
    ):
        """
        The linear sky light profile, representing the background sky emission as a constant sheet of values.

        For a positive only linear solver, a single sky profile cannot be used to solve for the sky background robustly.
        This is because the solution must be positive, meaning that negative solutions are not accessible.

        To address this, the sky is represented by two flat sky profiles, one with a positive intensity and one with a
        negative intensity. The positive linear solver then fits for the both the positive and negative sky values.
        The solution will have one of these values as zero, and the other as the sky background value.

        When a positive-negative solver is used, no loss of performance is incurred by using two profiles, even though
        the two solutions are fully degenerate. The same API is therefore used for both solvers, for convenience.
        """
        super().__init__(light_profile_list=[SkyPos(), SkyNeg()])


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
