from typing import Tuple

from autogalaxy.profiles import mass_profiles as mp


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
        relation: MassLightRelation = MassLightRelation(),
        luminosity: float = 1.0,
        centre: Tuple[float, float] = (0.0, 0.0),
    ):

        self.luminosity = luminosity
        self.relation = relation

        einstein_radius = relation.einstein_radius_from(luminosity=luminosity)

        super().__init__(centre=centre, einstein_radius=einstein_radius)


class EllIsothermalMLR(mp.EllIsothermal):
    def __init__(
        self,
        relation: MassLightRelation = MassLightRelation(),
        luminosity: float = 1.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
    ):

        self.luminosity = luminosity
        self.relation = relation

        einstein_radius = relation.einstein_radius_from(luminosity=luminosity)

        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            einstein_radius=einstein_radius,
        )
