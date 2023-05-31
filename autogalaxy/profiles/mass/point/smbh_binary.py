import numpy as np
from typing import Tuple

import autoarray as aa

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
            The (y,x) arc-second coordinates of centre of the SMBH binary, defined as the mid-point between the
            two SMBHs.
        separation
            The arc-second separation between the two SMBHs.
        angle_binary
            The angle between the two SMBHs relative to the positive x-axis of the centre of the SMBH binary.
        mass
            The sum of the masses of the two SMBHs in solar masses.
        mass_ratio
            The ratio of the mass of the second SMBH to the first SMBH. A mass ratio of 2.0 gives two SMBHs where
            the first SMBH has twice the mass of the second SMBH.
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
    def angle_binary_radians(self) -> float:
        """
        The angle between the two SMBHs in radians.
        """
        return self.angle_binary * np.pi / 180.0

    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The convergence is computed as the sum of the convergence of the two individual `SMBH` profiles in the binary.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """
        return self.smbh_0.convergence_2d_from(
            grid=grid
        ) + self.smbh_1.convergence_2d_from(grid=grid)

    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Returns the two dimensional projected potential on a grid of (y,x) arc-second coordinates.

        The potential is computed as the sum of the potential of the two individual `SMBH` profiles in the binary.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the potential is computed on.
        """
        return self.smbh_0.potential_2d_from(grid=grid) + self.smbh_1.potential_2d_from(
            grid=grid
        )

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Returns the two dimensional deflection angles on a grid of (y,x) arc-second coordinates.

        The deflection angles are computed as the sum of the convergence of the two individual `SMBH` profiles in the
        binary.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """
        return self.smbh_0.deflections_yx_2d_from(
            grid=grid
        ) + self.smbh_1.deflections_yx_2d_from(grid=grid)
