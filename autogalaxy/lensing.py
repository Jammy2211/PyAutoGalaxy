from autoconf import conf
import numpy as np
from autoarray.structures import arrays, grids
from autoarray.util import array_util
from scipy.optimize import root_scalar
from skimage import measure
from functools import wraps


def precompute_jacobian(func):
    @wraps(func)
    def wrapper(lensing_obj, grid, jacobian=None):

        if jacobian is None:
            jacobian = lensing_obj.jacobian_from_grid(grid=grid)

        return func(lensing_obj, grid, jacobian)

    return wrapper


class LensingObject:

    _preload_critical_curves = None
    _preload_caustics = None

    @property
    def mass_profiles(self):
        raise NotImplementedError("mass profiles list should be overriden")

    def convergence_func(self, grid_radius):
        raise NotImplementedError("convergence_func should be overridden")

    def convergence_from_grid(self, grid):
        raise NotImplementedError("convergence_from_grid should be overridden")

    def potential_func(self, u, y, x):
        raise NotImplementedError("potential_func should be overridden")

    def potential_from_grid(self, grid):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement potential_from_grid"
        )

    def deflections_from_grid(self, grid):
        raise NotImplementedError("deflections_from_grid should be overridden")

    @property
    def mass_profile_centres(self):
        raise NotImplementedError("mass profile centres should be overridden")

    def mass_integral(self, x):
        """Routine to integrate an elliptical light profiles - set axis ratio to 1 to compute the luminosity within a \
        circle"""
        return 2 * np.pi * x * self.convergence_func(grid_radius=x)

    def deflection_magnitudes_from_grid(self, grid):
        deflections = self.deflections_from_grid(grid=grid)
        return deflections.distances_from_coordinate(coordinate=(0.0, 0.0))

    def deflections_via_potential_from_grid(self, grid):

        potential = self.potential_from_grid(grid=grid)

        deflections_y_2d = np.gradient(potential.in_2d, grid.in_2d[:, 0, 0], axis=0)
        deflections_x_2d = np.gradient(potential.in_2d, grid.in_2d[0, :, 1], axis=1)

        return grids.Grid.manual_mask(
            grid=np.stack((deflections_y_2d, deflections_x_2d), axis=-1), mask=grid.mask
        )

    def jacobian_from_grid(self, grid):

        deflections = self.deflections_from_grid(grid=grid)

        a11 = arrays.Array.manual_mask(
            array=1.0
            - np.gradient(deflections.in_2d[:, :, 1], grid.in_2d[0, :, 1], axis=1),
            mask=grid.mask,
        )

        a12 = arrays.Array.manual_mask(
            array=-1.0
            * np.gradient(deflections.in_2d[:, :, 1], grid.in_2d[:, 0, 0], axis=0),
            mask=grid.mask,
        )

        a21 = arrays.Array.manual_mask(
            array=-1.0
            * np.gradient(deflections.in_2d[:, :, 0], grid.in_2d[0, :, 1], axis=1),
            mask=grid.mask,
        )

        a22 = arrays.Array.manual_mask(
            array=1
            - np.gradient(deflections.in_2d[:, :, 0], grid.in_2d[:, 0, 0], axis=0),
            mask=grid.mask,
        )

        return [[a11, a12], [a21, a22]]

    @precompute_jacobian
    def convergence_via_jacobian_from_grid(self, grid, jacobian=None):

        convergence = 1 - 0.5 * (jacobian[0][0] + jacobian[1][1])

        return arrays.Array(array=convergence, mask=grid.mask)

    @precompute_jacobian
    def shear_via_jacobian_from_grid(self, grid, jacobian=None):

        gamma_y = -0.5 * (jacobian[0][1] + jacobian[1][0])
        gamma_x = 0.5 * (jacobian[1][1] - jacobian[0][0])

        return arrays.Array(array=(gamma_x ** 2 + gamma_y ** 2) ** 0.5, mask=grid.mask)

    @precompute_jacobian
    def tangential_eigen_value_from_grid(self, grid, jacobian=None):

        convergence = self.convergence_via_jacobian_from_grid(
            grid=grid, jacobian=jacobian
        )

        shear = self.shear_via_jacobian_from_grid(grid=grid, jacobian=jacobian)

        return arrays.Array(array=1 - convergence - shear, mask=grid.mask)

    @precompute_jacobian
    def radial_eigen_value_from_grid(self, grid, jacobian=None):

        convergence = self.convergence_via_jacobian_from_grid(
            grid=grid, jacobian=jacobian
        )

        shear = self.shear_via_jacobian_from_grid(grid=grid, jacobian=jacobian)

        return arrays.Array(array=1 - convergence + shear, mask=grid.mask)

    def magnification_from_grid(self, grid):

        jacobian = self.jacobian_from_grid(grid=grid)

        det_jacobian = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]

        return arrays.Array(array=1 / det_jacobian, mask=grid.mask)

    def magnification_irregular_from_grid(self, grid, buffer=0.01):

        grid_shift_y_up = np.zeros(grid.shape)
        grid_shift_y_up[:, 0] = grid[:, 0] + buffer
        grid_shift_y_up[:, 1] = grid[:, 1]

        grid_shift_y_down = np.zeros(grid.shape)
        grid_shift_y_down[:, 0] = grid[:, 0] - buffer
        grid_shift_y_down[:, 1] = grid[:, 1]

        grid_shift_x_left = np.zeros(grid.shape)
        grid_shift_x_left[:, 0] = grid[:, 0]
        grid_shift_x_left[:, 1] = grid[:, 1] - buffer

        grid_shift_x_right = np.zeros(grid.shape)
        grid_shift_x_right[:, 0] = grid[:, 0]
        grid_shift_x_right[:, 1] = grid[:, 1] + buffer

        deflections_up = self.deflections_from_grid(grid=grid_shift_y_up)
        deflections_down = self.deflections_from_grid(grid=grid_shift_y_down)
        deflections_left = self.deflections_from_grid(grid=grid_shift_x_left)
        deflections_right = self.deflections_from_grid(grid=grid_shift_x_right)

        shear_yy = 0.5 * (deflections_up[:, 0] - deflections_down[:, 0]) / buffer
        shear_xy = 0.5 * (deflections_up[:, 1] - deflections_down[:, 1]) / buffer
        shear_yx = 0.5 * (deflections_right[:, 0] - deflections_left[:, 0]) / buffer
        shear_xx = 0.5 * (deflections_right[:, 1] - deflections_left[:, 1]) / buffer

        det_A = (1 - shear_xx) * (1 - shear_yy) - shear_xy * shear_yx

        return grid.values_from_arr_1d(arr_1d=1.0 / det_A)

    def tangential_critical_curve_from_grid(self, grid):

        tangential_eigen_values = self.tangential_eigen_value_from_grid(grid=grid)

        tangential_critical_curve_indices = measure.find_contours(
            tangential_eigen_values.in_2d, 0
        )

        if len(tangential_critical_curve_indices) == 0:
            return []

        tangential_critical_curve = grid.geometry.grid_scaled_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=tangential_critical_curve_indices[0],
            shape_2d=tangential_eigen_values.sub_shape_2d,
        )

        try:
            return grids.GridIrregularGrouped(tangential_critical_curve)
        except IndexError:
            return []

    def radial_critical_curve_from_grid(self, grid):

        radial_eigen_values = self.radial_eigen_value_from_grid(grid=grid)

        radial_critical_curve_indices = measure.find_contours(
            radial_eigen_values.in_2d, 0
        )

        if len(radial_critical_curve_indices) == 0:
            return []

        radial_critical_curve = grid.geometry.grid_scaled_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=radial_critical_curve_indices[0],
            shape_2d=radial_eigen_values.sub_shape_2d,
        )

        return grids.GridIrregularGrouped(radial_critical_curve)

    def critical_curves_from_grid(self, grid):

        if self._preload_critical_curves is not None:
            return self._preload_critical_curves

        if len(self.mass_profiles) == 0:
            return []

        try:
            return grids.GridIrregularGrouped(
                [
                    self.tangential_critical_curve_from_grid(grid=grid),
                    self.radial_critical_curve_from_grid(grid=grid),
                ]
            )
        except ValueError:
            return []

    def tangential_caustic_from_grid(self, grid):

        tangential_critical_curve = self.tangential_critical_curve_from_grid(grid=grid)

        if len(tangential_critical_curve) == 0:
            return []

        deflections_critical_curve = self.deflections_from_grid(
            grid=tangential_critical_curve
        )

        return tangential_critical_curve - deflections_critical_curve

    def radial_caustic_from_grid(self, grid):

        radial_critical_curve = self.radial_critical_curve_from_grid(grid=grid)

        if len(radial_critical_curve) == 0:
            return []

        deflections_critical_curve = self.deflections_from_grid(
            grid=radial_critical_curve
        )

        return radial_critical_curve - deflections_critical_curve

    def caustics_from_grid(self, grid):

        if self._preload_caustics is not None:
            return self._preload_caustics

        if len(self.mass_profiles) == 0:
            return []

        try:
            return grids.GridIrregularGrouped(
                [
                    self.tangential_caustic_from_grid(grid=grid),
                    self.radial_caustic_from_grid(grid=grid),
                ]
            )
        except IndexError:
            return []

    @array_util.Memoizer()
    def area_within_tangential_critical_curve_from_grid(self, grid):

        tangential_critical_curve = self.tangential_critical_curve_from_grid(grid=grid)
        x, y = tangential_critical_curve[:, 0], tangential_critical_curve[:, 1]

        return np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y)))

    def einstein_radius_via_tangential_critical_curve_from_grid(self, grid):
        return np.sqrt(
            self.area_within_tangential_critical_curve_from_grid(grid=grid) / np.pi
        )

    def einstein_mass_angular_via_tangential_critical_curve_from_grid(self, grid):
        return np.pi * (
            self.einstein_radius_via_tangential_critical_curve_from_grid(grid=grid) ** 2
        )
