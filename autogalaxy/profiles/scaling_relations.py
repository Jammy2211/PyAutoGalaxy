from autogalaxy.profiles import mass_profiles as mp

from typing import Tuple


class MassLightRelation:
    def __init__(self, gradient=1.0, denominator=1.0, power=0.5):

        self.gradient = gradient
        self.denominator = denominator
        self.power = power

    def einstein_radius_from(self, luminosity):

        return self.gradient * ((luminosity / self.denominator) ** self.power)


class SphIsothermalMLR(mp.SphIsothermal):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        luminosity: float = 1.0,
        relation: MassLightRelation = MassLightRelation(),
    ):

        self.luminosity = luminosity
        self.relation = relation

        einstein_radius = relation.einstein_radius_from(luminosity=luminosity)

        super().__init__(centre=centre, einstein_radius=einstein_radius)
