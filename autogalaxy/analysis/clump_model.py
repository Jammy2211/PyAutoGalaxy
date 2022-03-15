from typing import List, Optional, Type

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
        """
        The clump API allows creates model components which model the light and mass of galaxies that are nearby the 
        main galaxy(s) of interest. 
        
        The `ClumpModel` object handles the creation of these model components to streamline model composition with
        multiple clumps.
        
        Every galaxy which is modeled as a clump has its centre input into this object which is fixed to this value
        for model-fitting. All clumps are created as model `Galaxy` objects with a shard input redshift.
        
        The light and mass profiles of the clumps are input via the `light_cls` and `mass_cls` inputs. If either is
        omitted the clumps are not assigned a light or mass model.
        
        Parameters
        ----------
        redshift
            The redshift value of all clumps, which is likely the same as the main galaxy redshift.
        centres
            The centre of every clump in the model, whose light and mass profile centres are fixed to this value 
            throughout the model-fit.
        light_cls
            The light profile given too all clumps; if omitted all clumps have no light profile.
        mass_cls
            The mass profile given too all clumps; if omitted all clumps have no mass profile.
        einstein_radius_upper_limit
            The upper limit given to any mass model's `einstein_radius` parameter (e.g. if `SphIsothermal` profiles
            are used to model clumps).
        """
        self.redshift = redshift
        self.centres = centres

        self.light_cls = light_cls
        self.mass_cls = mass_cls

        self.einstein_radius_upper_limit = einstein_radius_upper_limit

    @property
    def total_clumps(self) -> int:
        return len(self.centres.in_list)

    @property
    def light_list(self) -> Optional[List[af.Model]]:
        """
        Returns a list of every clump's light model, where the centre of that light model is fixed to its corresponding 
        input clump's centre.
        """
        if self.light_cls is None:
            return None

        return [
            af.Model(self.light_cls, centre=centre) for centre in self.centres.in_list
        ]

    @property
    def mass_list(self) -> Optional[List[af.Model]]:
        """
        Returns a list of every clump's mass model, where the centre of that mass model is fixed to its corresponding 
        input clump's centre.
        """
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
    def clumps_light_only(self) -> af.Collection:
        """
        Returns all clumps as a `Collection` model object, which can be easily added to an overall model `Collection`
        of galaxies (see the `clumps` property below for more details).

        All clumps only contain their model mass profiles, which is important for certain pipelines which omit
        the clump mass profiles.
        """
        clumps_light_only = {}

        for i in range(self.total_clumps):

            light = self.light_list[i] if self.light_cls is not None else None

            clumps_light_only[f"clump_{i}"] = af.Model(
                Galaxy, redshift=self.redshift, light=light
            )

        return af.Collection(**clumps_light_only)

    @property
    def clumps_mass_only(self) -> af.Collection:
        """
        Returns all clumps as a `Collection` model object, which can be easily added to an overall model `Collection`
        of galaxies (see the `clumps` property below for more details).

        All clumps only contain their model mass profiles, which is important for certain pipelines which omit
        the clump light profiles.
        """
        clumps_mass_only = {}

        for i in range(self.total_clumps):

            mass = self.mass_list[i] if self.mass_cls is not None else None

            clumps_mass_only[f"clump_{i}"] = af.Model(
                Galaxy, redshift=self.redshift, mass=mass
            )

        return af.Collection(**clumps_mass_only)

    @property
    def clumps(self) -> af.Collection:
        """
        Returns all clumps as a `Collection` model object, which can be easily added to an overall model `Collection`
        of galaxies.

        To make this `Collection` every clump centre and the input `light_cls` / `mass_cls` are used to create a
        dictionary of model `Galaxy` objects with fixed light and mass profile centres. Their redshifts use the
        input redshift.

        The keys of this dictionary are numerically ordered as `clump_0`, `clump_2` etc.
        """
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
        return af.Collection()

    @property
    def clumps_mass_only(self):
        return af.Collection()

    @property
    def clumps(self):
        return af.Collection()
