from autogalaxy.profiles import mass_profiles as mp

from typing import Tuple


class MassLightRelation:
    def __init__(self, gradient=1.0, intercept=1.0):

        self.gradient = gradient
        self.intercept = intercept

    def mass_from(self, magnitude):

        return magnitude * self.gradient + self.intercept


class SphIsothermalMLR(mp.SphIsothermal):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        magnitude: float = 1.0,
        relation: MassLightRelation = MassLightRelation(),
    ):

        self.magnitude = magnitude
        self.relation = relation

        self.mass = relation.mass_from(magnitude=magnitude)

        einstein_radius = self.mass

        super().__init__(centre=centre, einstein_radius=einstein_radius)
