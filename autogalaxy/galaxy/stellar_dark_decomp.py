from autogalaxy import exc
from autogalaxy.profiles.mass.dark.abstract import DarkProfile
from autogalaxy.profiles.mass.stellar.abstract import StellarProfile


class StellarDarkDecomp:
    def __init__(self, galaxy):
        self.galaxy = galaxy

    def stellar_mass_angular_within_circle_from(self, radius: float):
        if self.galaxy.has(cls=StellarProfile):
            return sum(
                [
                    profile.mass_angular_within_circle_from(radius=radius)
                    for profile in self.galaxy.cls_list_from(cls=StellarProfile)
                ]
            )
        else:
            raise exc.GalaxyException(
                "You cannot perform a stellar mass-based calculation on a galaxy which does not have a stellar "
                "mass-profile "
            )

    def dark_mass_angular_within_circle_from(self, radius: float):
        if self.galaxy.has(cls=DarkProfile):
            return sum(
                [
                    profile.mass_angular_within_circle_from(radius=radius)
                    for profile in self.galaxy.cls_list_from(cls=DarkProfile)
                ]
            )
        else:
            raise exc.GalaxyException(
                "You cannot perform a dark mass-based calculation on a galaxy which does not have a dark mass-profile"
            )

    def stellar_fraction_at_radius_from(self, radius):
        return 1.0 - self.dark_fraction_at_radius_from(radius=radius)

    def dark_fraction_at_radius_from(self, radius):
        stellar_mass = self.stellar_mass_angular_within_circle_from(radius=radius)
        dark_mass = self.dark_mass_angular_within_circle_from(radius=radius)

        return dark_mass / (stellar_mass + dark_mass)
