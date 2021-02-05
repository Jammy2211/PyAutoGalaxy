import numpy as np
from autoarray.structures import arrays, grids
from autoarray.util import array_util
from skimage import measure
from functools import wraps


def precompute_jacobian(func):
    @wraps(func)
    def wrapper(lensing_obj, grid, jacobian=None):

        if jacobian is None:
            jacobian = lensing_obj.jacobian_from_grid(grid=grid)

        return func(lensing_obj, grid, jacobian)

    return wrapper


def evaluation_grid(func):
    @wraps(func)
    def wrapper(lensing_obj, grid, pixel_scale=0.05):

        if hasattr(grid, "is_evaluation_grid"):
            if grid.is_evaluation_grid:
                return func(lensing_obj, grid, pixel_scale)

        pixel_scale_ratio = grid.pixel_scale / pixel_scale

        zoom_shape_native = grid.mask.zoom_shape_native
        shape_native = (
            int(pixel_scale_ratio * zoom_shape_native[0]),
            int(pixel_scale_ratio * zoom_shape_native[1]),
        )

        grid = grids.Grid2D.uniform(
            shape_native=shape_native,
            pixel_scales=(pixel_scale, pixel_scale),
            origin=grid.mask.zoom_offset_scaled,
        )

        grid.is_evaluation_grid = True

        return func(lensing_obj, grid, pixel_scale)

    return wrapper


class LensingObject:
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

        deflections_y_2d = np.gradient(potential.native, grid.native[:, 0, 0], axis=0)
        deflections_x_2d = np.gradient(potential.native, grid.native[0, :, 1], axis=1)

        return grids.Grid2D.manual_mask(
            grid=np.stack((deflections_y_2d, deflections_x_2d), axis=-1), mask=grid.mask
        )

    def jacobian_from_grid(self, grid):

        deflections = self.deflections_from_grid(grid=grid)

        a11 = arrays.Array2D.manual_mask(
            array=1.0
            - np.gradient(deflections.native[:, :, 1], grid.native[0, :, 1], axis=1),
            mask=grid.mask,
        )

        a12 = arrays.Array2D.manual_mask(
            array=-1.0
            * np.gradient(deflections.native[:, :, 1], grid.native[:, 0, 0], axis=0),
            mask=grid.mask,
        )

        a21 = arrays.Array2D.manual_mask(
            array=-1.0
            * np.gradient(deflections.native[:, :, 0], grid.native[0, :, 1], axis=1),
            mask=grid.mask,
        )

        a22 = arrays.Array2D.manual_mask(
            array=1
            - np.gradient(deflections.native[:, :, 0], grid.native[:, 0, 0], axis=0),
            mask=grid.mask,
        )

        return [[a11, a12], [a21, a22]]

    @precompute_jacobian
    def convergence_via_jacobian_from_grid(self, grid, jacobian=None):

        convergence = 1 - 0.5 * (jacobian[0][0] + jacobian[1][1])

        return arrays.Array2D(array=convergence, mask=grid.mask)

    @precompute_jacobian
    def shear_via_jacobian_from_grid(self, grid, jacobian=None):

        shear_y = -0.5 * (jacobian[0][1] + jacobian[1][0])
        shear_x = 0.5 * (jacobian[1][1] - jacobian[0][0])

        return arrays.Array2D(
            array=(shear_x ** 2 + shear_y ** 2) ** 0.5, mask=grid.mask
        )

    @precompute_jacobian
    def tangential_eigen_value_from_grid(self, grid, jacobian=None):

        convergence = self.convergence_via_jacobian_from_grid(
            grid=grid, jacobian=jacobian
        )

        shear = self.shear_via_jacobian_from_grid(grid=grid, jacobian=jacobian)

        return arrays.Array2D(array=1 - convergence - shear, mask=grid.mask)

    @precompute_jacobian
    def radial_eigen_value_from_grid(self, grid, jacobian=None):

        convergence = self.convergence_via_jacobian_from_grid(
            grid=grid, jacobian=jacobian
        )

        shear = self.shear_via_jacobian_from_grid(grid=grid, jacobian=jacobian)

        return arrays.Array2D(array=1 - convergence + shear, mask=grid.mask)

    def magnification_from_grid(self, grid):

        jacobian = self.jacobian_from_grid(grid=grid)

        det_jacobian = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]

        return arrays.Array2D(array=1 / det_jacobian, mask=grid.mask)

    def hessian_from_grid(self, grid, buffer=0.01):

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

        hessian_yy = 0.5 * (deflections_up[:, 0] - deflections_down[:, 0]) / buffer
        hessian_xy = 0.5 * (deflections_up[:, 1] - deflections_down[:, 1]) / buffer
        hessian_yx = 0.5 * (deflections_right[:, 0] - deflections_left[:, 0]) / buffer
        hessian_xx = 0.5 * (deflections_right[:, 1] - deflections_left[:, 1]) / buffer

        return hessian_yy, hessian_xy, hessian_yx, hessian_xx

    def convergence_via_hessian_from_grid(self, grid, buffer=0.01):

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from_grid(
            grid=grid, buffer=buffer
        )

        return grid.values_from_array_slim(array_slim=0.5 * (hessian_yy + hessian_xx))

    def shear_via_hessian_from_grid(self, grid, buffer=0.01):

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from_grid(
            grid=grid, buffer=buffer
        )

        shear_y = 0.5 * (hessian_xx - hessian_yy)
        shear_x = hessian_xy

        return grid.values_from_array_slim(
            array_slim=(shear_x ** 2 + shear_y ** 2) ** 0.5
        )

    def magnification_via_hessian_from_grid(self, grid, buffer=0.01):

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from_grid(
            grid=grid, buffer=buffer
        )

        det_A = (1 - hessian_xx) * (1 - hessian_yy) - hessian_xy * hessian_yx

        return grid.values_from_array_slim(array_slim=1.0 / det_A)

    @evaluation_grid
    def tangential_critical_curve_from_grid(self, grid, pixel_scale=0.05):

        tangential_eigen_values = self.tangential_eigen_value_from_grid(grid=grid)

        tangential_critical_curve_indices = measure.find_contours(
            tangential_eigen_values.native, 0
        )

        if len(tangential_critical_curve_indices) == 0:
            return []

        tangential_critical_curve = grid.mask.grid_scaled_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=tangential_critical_curve_indices[0],
            shape_native=tangential_eigen_values.sub_shape_native,
        )

        try:
            return grids.Grid2DIrregularGrouped(tangential_critical_curve)
        except IndexError:
            return []

    @evaluation_grid
    def radial_critical_curve_from_grid(self, grid, pixel_scale=0.05):

        radial_eigen_values = self.radial_eigen_value_from_grid(grid=grid)

        radial_critical_curve_indices = measure.find_contours(
            radial_eigen_values.native, 0
        )

        if len(radial_critical_curve_indices) == 0:
            return []

        radial_critical_curve = grid.mask.grid_scaled_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=radial_critical_curve_indices[0],
            shape_native=radial_eigen_values.sub_shape_native,
        )

        try:
            return grids.Grid2DIrregularGrouped(radial_critical_curve)
        except IndexError:
            return []

    @evaluation_grid
    def critical_curves_from_grid(self, grid, pixel_scale=0.05):

        if len(self.mass_profiles) == 0:
            return []

        try:
            return grids.Grid2DIrregularGrouped(
                [
                    self.tangential_critical_curve_from_grid(
                        grid=grid, pixel_scale=pixel_scale
                    ),
                    self.radial_critical_curve_from_grid(
                        grid=grid, pixel_scale=pixel_scale
                    ),
                ]
            )
        except (IndexError, ValueError):
            return []

    @evaluation_grid
    def tangential_caustic_from_grid(self, grid, pixel_scale=0.05):

        tangential_critical_curve = self.tangential_critical_curve_from_grid(
            grid=grid, pixel_scale=pixel_scale
        )

        if len(tangential_critical_curve) == 0:
            return []

        deflections_critical_curve = self.deflections_from_grid(
            grid=tangential_critical_curve
        )

        return tangential_critical_curve - deflections_critical_curve

    @evaluation_grid
    def radial_caustic_from_grid(self, grid, pixel_scale=0.05):

        radial_critical_curve = self.radial_critical_curve_from_grid(
            grid=grid, pixel_scale=pixel_scale
        )

        if len(radial_critical_curve) == 0:
            return []

        deflections_critical_curve = self.deflections_from_grid(
            grid=radial_critical_curve
        )

        return radial_critical_curve - deflections_critical_curve

    @evaluation_grid
    def caustics_from_grid(self, grid, pixel_scale=0.05):

        if len(self.mass_profiles) == 0:
            return []

        try:
            return grids.Grid2DIrregularGrouped(
                [
                    self.tangential_caustic_from_grid(
                        grid=grid, pixel_scale=pixel_scale
                    ),
                    self.radial_caustic_from_grid(grid=grid, pixel_scale=pixel_scale),
                ]
            )
        except (IndexError, ValueError):
            return []

    @evaluation_grid
    def area_within_tangential_critical_curve_from_grid(self, grid, pixel_scale=0.05):

        tangential_critical_curve = self.tangential_critical_curve_from_grid(
            grid=grid, pixel_scale=pixel_scale
        )
        x, y = tangential_critical_curve[:, 0], tangential_critical_curve[:, 1]

        return np.abs(0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y)))

    @evaluation_grid
    def einstein_radius_from_grid(self, grid, pixel_scale=0.05):

        try:
            return np.sqrt(
                self.area_within_tangential_critical_curve_from_grid(
                    grid=grid, pixel_scale=pixel_scale
                )
                / np.pi
            )
        except TypeError:
            raise TypeError("The grid input was unable to estimate the Einstein Radius")

    @evaluation_grid
    def einstein_mass_angular_from_grid(self, grid, pixel_scale=0.05):
        return np.pi * (
            self.einstein_radius_from_grid(grid=grid, pixel_scale=pixel_scale) ** 2
        )
