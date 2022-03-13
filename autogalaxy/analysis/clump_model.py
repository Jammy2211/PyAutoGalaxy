from typing import Type, Optional

import autofit as af
import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.profiles.mass_profiles.mass_profiles import MassProfile


class ClumpModel:
    def __init__(
        self,
        redshift: float,
        centres: aa.Grid2DIrregular,
        light_cls: Optional[Type[LightProfile]] = None,
        mass_cls: Optional[Type[MassProfile]] = None,
        einstein_radius_upper_limit: Optional[float] = None,
    ):

        self.redshift = redshift
        self.centres = centres

        self.light_cls = light_cls
        self.mass_cls = mass_cls

        self.einstein_radius_upper_limit = einstein_radius_upper_limit

    @property
    def total_clumps(self):
        return len(self.centres.in_list)

    @property
    def light_list(self):

        if self.light_cls is None:
            return None

        return [
            af.Model(self.light_cls, centre=centre) for centre in self.centres.in_list
        ]

    @property
    def mass_list(self):

        if self.mass_cls is None:
            return None

        mass_list = []

        for centre in self.centres.in_list:

            mass = af.Model(self.mass_cls, centre=centre)

            if (
                hasattr(mass, "einstein_radius")
                and self.einstein_radius_upper_limit is not None
            ):
                mass.einstein_radius = af.UniformPrior(
                    lower_limit=0.0, upper_limit=self.einstein_radius_upper_limit
                )

            mass_list.append(mass)

        return mass_list

    @property
    def clumps_light_only(self):

        clumps_light_only = {}

        for i in range(self.total_clumps):

            light = self.light_list[i] if self.light_cls is not None else None

            clumps_light_only[f"clump_{i}"] = af.Model(
                Galaxy, redshift=self.redshift, light=light
            )

        return clumps_light_only

    @property
    def clumps_mass_only(self):

        clumps_mass_only = {}

        for i in range(self.total_clumps):

            mass = self.mass_list[i] if self.mass_cls is not None else None

            clumps_mass_only[f"clump_{i}"] = af.Model(
                Galaxy, redshift=self.redshift, mass=mass
            )

        return clumps_mass_only

    @property
    def clumps(self):

        clumps = {}

        for i in range(self.total_clumps):

            light = self.light_list[i] if self.light_cls is not None else None
            mass = self.mass_list[i] if self.mass_cls is not None else None

            clumps[f"clump_{i}"] = af.Model(
                Galaxy, redshift=self.redshift, light=light, mass=mass
            )

        return af.Collection(**clumps)


class ClumpModelDisabled:
    def __init__(self):

        pass

    @property
    def clumps_light_only(self):
        return {}

    @property
    def clumps_mass_only(self):
        return {}

    @property
    def clumps(self):
        return {}
