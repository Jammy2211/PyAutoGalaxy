import numpy as np
from typing import Tuple

from autogalaxy.profiles.mass.abstract.abstract import MassProfile
from autogalaxy.profiles.mass.point.smbh import SMBH


class SMBHBinary(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        separation: float = 1.0,
        angle_binary: float = 0.0,
        mass: float = 1e10,
        mass_ratio: float = 1.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        """
        Represents a supermassive black hole (SMBH) binary (e.g. two merging SMBH's at the centre of a galaxy).

        This uses two `SMBH` mass profiles to represent the SMBHs.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        mass
            The mass of the SMBH in solar masses.
        redshift_object
            The redshift of the SMBH, which is used to convert its mass to an Einstein radius.
        redshift_source
            The redshift of the source galaxy, which is used to convert the mass of the SMBH to an Einstein radius.
        """

        self.separation = separation
        self.angle_binary = angle_binary
        self.mass = mass
        self.mass_ratio = mass_ratio
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        x_0 = centre[1] + (self.separation / 2.0) * np.cos(self.angle_binary_radians)
        y_0 = centre[0] + (self.separation / 2.0) * np.sin(self.angle_binary_radians)

        if mass_ratio >= 1.0:
            mass_0 = mass * (mass_ratio / mass)
        else:
            mass_0 = mass - mass * ((1.0 / mass_ratio) / mass)

        self.smbh_0 = SMBH(
            centre=(y_0, x_0),
            mass=mass_0,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        x_1 = centre[1] + (self.separation / 2.0) * np.cos(
            self.angle_binary_radians - np.pi
        )
        y_1 = centre[0] + (self.separation / 2.0) * np.sin(
            self.angle_binary_radians - np.pi
        )

        mass_1 = mass - mass_0

        self.smbh_1 = SMBH(
            centre=(y_1, x_1),
            mass=mass_1,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(centre=centre, ell_comps=(0.0, 0.0))

    @property
    def angle_binary_radians(self):
        return self.angle_binary * np.pi / 180.0
