import numpy as np

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile

from autogalaxy import convert


class ExternalShear(MassProfile):
    def __init__(self, gamma_1: float = 0.0, gamma_2: float = 0.0):
        """
        An `ExternalShear` term, to model the line-of-sight contribution of other galaxies / satellites.

        The shear angle is defined in the direction of stretching of the image. Therefore, if an object located \
        outside the lens is responsible for the shear, it will be offset 90 degrees from the value of angle.

        Parameters
        ----------
        gamma
        """

        super().__init__(centre=(0.0, 0.0), ell_comps=(0.0, 0.0))
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    @property
    def magnitude(self):
        return convert.shear_magnitude_from(gamma_1=self.gamma_1, gamma_2=self.gamma_2)

    @property
    def angle(self):
        return convert.shear_angle_from(gamma_1=self.gamma_1, gamma_2=self.gamma_2)

    def convergence_func(self, grid_radius: float) -> float:
        return 0.0

    def average_convergence_of_1_radius(self):
        return 0.0

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        shear_angle = (
            self.angle - 90
        )  ##to be onsistent with autolens deflection angle calculation
        phig = np.deg2rad(shear_angle)
        shear_amp = self.magnitude
        phicoord = np.arctan2(grid[:, 0], grid[:, 1])
        rcoord = np.sqrt(grid[:, 0] ** 2.0 + grid[:, 1] ** 2.0)

        return -0.5 * shear_amp * rcoord**2 * np.cos(2 * (phicoord - phig))

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        deflection_y = -np.multiply(self.magnitude, grid[:, 0])
        deflection_x = np.multiply(self.magnitude, grid[:, 1])
        return self.rotated_grid_from_reference_frame_from(
            np.vstack((deflection_y, deflection_x)).T
        )
