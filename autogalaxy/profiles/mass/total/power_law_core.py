import copy
import numpy as np
from scipy.integrate import quad
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class PowerLawCore(MassProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored elliptical power-law density distribution

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(centre=centre, ell_comps=ell_comps)

        self.einstein_radius = einstein_radius
        self.slope = slope
        self.core_radius = core_radius

    @property
    def einstein_radius_rescaled(self):
        """
        Rescale the einstein radius by slope and axis_ratio, to reduce its degeneracy with other mass-profiles
        parameters.
        """
        return ((3 - self.slope) / (1 + self.axis_ratio)) * self.einstein_radius ** (
            self.slope - 1
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Returns the two dimensional projected convergence on a grid of (y,x) arc-second coordinates.

        The `grid_2d_to_structure` decorator reshapes the ndarrays the convergence is outputted on. See
        *aa.grid_2d_to_structure* for a description of the output.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.
        """

        covnergence_grid = np.zeros(grid.shape[0])

        grid_eta = self.elliptical_radii_grid_from(grid)

        for i in range(grid.shape[0]):
            covnergence_grid[i] = self.convergence_func(grid_eta[i])

        return covnergence_grid

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the potential on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.
        """

        potential_grid = np.zeros(grid.shape[0])

        for i in range(grid.shape[0]):
            potential_grid[i] = quad(
                self.potential_func,
                a=0.0,
                b=1.0,
                args=(
                    grid[i, 0],
                    grid[i, 1],
                    self.axis_ratio,
                    self.slope,
                    self.core_radius,
                ),
            )[0]

        return self.einstein_radius_rescaled * self.axis_ratio * potential_grid

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        def calculate_deflection_component(npow, index):
            einstein_radius_rescaled = self.einstein_radius_rescaled

            deflection_grid = self.axis_ratio * grid[:, index]

            for i in range(grid.shape[0]):
                deflection_grid[i] *= (
                    einstein_radius_rescaled
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid[i, 0],
                            grid[i, 1],
                            npow,
                            self.axis_ratio,
                            self.slope,
                            self.core_radius,
                        ),
                    )[0]
                )

            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotated_grid_from_reference_frame_from(
            grid=np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    def convergence_func(self, grid_radius: float) -> float:
        return self.einstein_radius_rescaled * (
            self.core_radius**2 + grid_radius**2
        ) ** (-(self.slope - 1) / 2.0)

    @staticmethod
    def potential_func(u, y, x, axis_ratio, slope, core_radius):
        eta = np.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        return (
            (eta / u)
            * ((3.0 - slope) * eta) ** -1.0
            * (
                (core_radius**2.0 + eta**2.0) ** ((3.0 - slope) / 2.0)
                - core_radius ** (3 - slope)
            )
            / ((1 - (1 - axis_ratio**2) * u) ** 0.5)
        )

    @staticmethod
    def deflection_func(u, y, x, npow, axis_ratio, slope, core_radius):
        _eta_u = np.sqrt((u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u)))))
        return (core_radius**2 + _eta_u**2) ** (-(slope - 1) / 2.0) / (
            (1 - (1 - axis_ratio**2) * u) ** (npow + 0.5)
        )

    @property
    def ellipticity_rescale(self):
        return (1.0 + self.axis_ratio) / 2.0

    @property
    def unit_mass(self):
        return "angular"


class PowerLawCoreSph(PowerLawCore):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        einstein_radius: float = 1.0,
        slope: float = 2.0,
        core_radius: float = 0.01,
    ):
        """
        Represents a cored spherical power-law density distribution

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius.
        slope
            The density slope of the power-law (lower value -> shallower profile, higher value -> steeper profile).
        core_radius
            The arc-second radius of the inner core.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            einstein_radius=einstein_radius,
            slope=slope,
            core_radius=core_radius,
        )

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles on a grid of (y,x) arc-second coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """
        eta = self.radial_grid_from(grid=grid)
        deflection = np.multiply(
            2.0 * self.einstein_radius_rescaled,
            np.divide(
                np.add(
                    np.power(
                        np.add(self.core_radius**2, np.square(eta)),
                        (3.0 - self.slope) / 2.0,
                    ),
                    -self.core_radius ** (3 - self.slope),
                ),
                np.multiply((3.0 - self.slope), eta),
            ),
        )
        return self._cartesian_grid_via_radial_from(grid=grid, radius=deflection)
