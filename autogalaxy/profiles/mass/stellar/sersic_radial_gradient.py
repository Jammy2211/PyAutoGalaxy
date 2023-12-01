import numpy as np
from scipy.integrate import quad
from typing import List, Tuple

import autoarray as aa

from autogalaxy.profiles.mass.stellar.sersic import AbstractSersic

from autogalaxy.profiles.mass.stellar.sersic import cse_settings_from


class SersicRadialGradient(AbstractSersic):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        ell_comps
            The first and second ellipticity components of the elliptical coordinate system.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """
        super().__init__(
            centre=centre,
            ell_comps=ell_comps,
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
        )
        self.mass_to_light_gradient = mass_to_light_gradient

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_2d_via_integral_from(self, grid: aa.type.Grid2DLike):
        """
        Calculate the deflection angles at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the deflection angles are computed on.

        """

        def calculate_deflection_component(npow, index):
            sersic_constant = self.sersic_constant

            deflection_grid = self.axis_ratio * grid[:, index]

            for i in range(grid.shape[0]):
                deflection_grid[i] *= (
                    self.intensity
                    * self.mass_to_light_ratio
                    * quad(
                        self.deflection_func,
                        a=0.0,
                        b=1.0,
                        args=(
                            grid[i, 0],
                            grid[i, 1],
                            npow,
                            self.axis_ratio,
                            self.sersic_index,
                            self.effective_radius,
                            self.mass_to_light_gradient,
                            sersic_constant,
                        ),
                    )[0]
                )
            return deflection_grid

        deflection_y = calculate_deflection_component(1.0, 0)
        deflection_x = calculate_deflection_component(0.0, 1)

        return self.rotated_grid_from_reference_frame_from(
            np.multiply(1.0, np.vstack((deflection_y, deflection_x)).T)
        )

    @staticmethod
    def deflection_func(
        u,
        y,
        x,
        npow,
        axis_ratio,
        sersic_index,
        effective_radius,
        mass_to_light_gradient,
        sersic_constant,
    ):
        _eta_u = np.sqrt(axis_ratio) * np.sqrt(
            (u * ((x**2) + (y**2 / (1 - (1 - axis_ratio**2) * u))))
        )

        return (
            (((axis_ratio * _eta_u) / effective_radius) ** -mass_to_light_gradient)
            * np.exp(
                -sersic_constant
                * (((_eta_u / effective_radius) ** (1.0 / sersic_index)) - 1)
            )
            / ((1 - (1 - axis_ratio**2) * u) ** (npow + 0.5))
        )

    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        """Calculate the projected convergence at a given set of arc-second gridded coordinates.

        Parameters
        ----------
        grid
            The grid of (y,x) arc-second coordinates the convergence is computed on.

        """
        return self.convergence_func(self.eccentric_radii_grid_from(grid))

    def convergence_func(self, grid_radius: float) -> float:
        return (
            self.mass_to_light_ratio
            * (
                ((self.axis_ratio * grid_radius) / self.effective_radius)
                ** -self.mass_to_light_gradient
            )
            * self.image_2d_via_radii_from(grid_radius)
        )

    def decompose_convergence_via_mge(self):
        radii_min = self.effective_radius / 100.0
        radii_max = self.effective_radius * 20.0

        def sersic_radial_gradient_2D(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * (
                    ((self.axis_ratio * r) / self.effective_radius)
                    ** -self.mass_to_light_gradient
                )
                * np.exp(
                    -self.sersic_constant
                    * (((r / self.effective_radius) ** (1.0 / self.sersic_index)) - 1.0)
                )
            )

        return self._decompose_convergence_via_mge(
            func=sersic_radial_gradient_2D, radii_min=radii_min, radii_max=radii_max
        )

    def decompose_convergence_via_cse(
        self, grid_radii: np.ndarray
    ) -> Tuple[List, List]:
        """
        Decompose the convergence of the Sersic profile into singular isothermal elliptical (sie) profiles.

        This decomposition uses the standard 2d profile of a Sersic mass profile.

        Parameters
        ----------
        func
            The function representing the profile that is decomposed into CSEs.
        radii_min:
            The minimum radius to fit
        radii_max:
            The maximum radius to fit
        total_sies
            The number of SIEs used to approximate the input func.
        sample_points: int (should be larger than 'total_sies')
            The number of data points to fit

        Returns
        -------
        Tuple[List, List]
            A list of amplitudes and core radii of every singular isothernal ellipsoids (sie) the mass profile is decomposed
            into.
        """

        upper_dex, lower_dex, total_cses, sample_points = cse_settings_from(
            effective_radius=self.effective_radius,
            sersic_index=self.sersic_index,
            sersic_constant=self.sersic_constant,
            mass_to_light_gradient=self.mass_to_light_gradient,
        )

        scaled_effective_radius = self.effective_radius / np.sqrt(self.axis_ratio)
        radii_min = scaled_effective_radius / 10.0**lower_dex
        radii_max = scaled_effective_radius * 10.0**upper_dex

        def sersic_radial_gradient_2D(r):
            return (
                self.mass_to_light_ratio
                * self.intensity
                * (
                    ((self.axis_ratio * r) / scaled_effective_radius)
                    ** -self.mass_to_light_gradient
                )
                * np.exp(
                    -self.sersic_constant
                    * (
                        ((r / scaled_effective_radius) ** (1.0 / self.sersic_index))
                        - 1.0
                    )
                )
            )

        return self._decompose_convergence_via_cse_from(
            func=sersic_radial_gradient_2D,
            radii_min=radii_min,
            radii_max=radii_max,
            total_cses=total_cses,
            sample_points=sample_points,
        )


class SersicRadialGradientSph(SersicRadialGradient):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        intensity: float = 0.1,
        effective_radius: float = 0.6,
        sersic_index: float = 0.6,
        mass_to_light_ratio: float = 1.0,
        mass_to_light_gradient: float = 0.0,
    ):
        """
        Setup a Sersic mass and light profiles.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        intensity
            Overall flux intensity normalisation in the light profiles (electrons per second).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        mass_to_light_ratio
            The mass-to-light ratio of the light profile.
        mass_to_light_gradient
            The mass-to-light radial gradient.
        """
        super().__init__(
            centre=centre,
            ell_comps=(0.0, 0.0),
            intensity=intensity,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            mass_to_light_ratio=mass_to_light_ratio,
            mass_to_light_gradient=mass_to_light_gradient,
        )
