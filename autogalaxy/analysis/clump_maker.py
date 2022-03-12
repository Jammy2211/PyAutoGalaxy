from typing import Type, Optional

import autofit as af
import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.profiles.mass_profiles.mass_profiles import MassProfile


class ClumpMaker:
    def __init__(
        self,
        redshift: float,
        light_cls: Optional[Type[LightProfile]] = None,
        mass_cls: Optional[Type[MassProfile]] = None,
    ):

        self.redshift = redshift

        self.light_cls = light_cls
        self.mass_cls = mass_cls

    def clump_dict_from(self, centres: aa.Grid2DIrregular):

        clump_dict = {}

        for i, centre in enumerate(centres.in_list):

            if self.light_cls is not None:

                light = af.Model(self.light_cls)
                light.centre = centre

            else:

                light = None

            if self.mass_cls is not None:

                mass = af.Model(self.mass_cls)
                mass.centre = centre

            else:

                mass = None

            clump_dict[f"clump_{i}"] = af.Model(
                Galaxy, redshift=self.redshift, light=light, mass=mass
            )

        return clump_dict

    def light_clump_model_dict_from(self, centres: aa.Grid2DIrregular):

        return {}
