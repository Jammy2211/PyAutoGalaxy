import jax
from jax import jit
import jax.numpy as jnp
from functools import wraps, partial
import logging
from typing import List, Tuple, Union


import autoarray as aa

from autogalaxy.util.shear_field import ShearYX2D
from autogalaxy.util.shear_field import ShearYX2DIrregular

logger = logging.getLogger(__name__)


def grid_scaled_2d_for_marching_squares_from(
    grid_pixels_2d: aa.Grid2D,
    shape_native: Tuple[int, int],
    mask: aa.Mask2D,
) -> aa.Grid2DIrregular:
    pixel_scales = mask.pixel_scales
    origin = mask.origin

    grid_scaled_1d = aa.util.geometry.grid_scaled_2d_slim_from(
        grid_pixels_2d_slim=grid_pixels_2d,
        shape_native=shape_native,
        pixel_scales=pixel_scales,
        origin=origin,
    )

    grid_scaled_1d[:, 0] -= pixel_scales[0] / 2.0
    grid_scaled_1d[:, 1] += pixel_scales[1] / 2.0

    return aa.Grid2DIrregular(values=grid_scaled_1d)


def precompute_jacobian(func):
    @wraps(func)
    def wrapper(lensing_obj, grid, jacobian=None):
        if jacobian is None:
            jacobian = lensing_obj.jacobian_from(grid=grid)

        return func(lensing_obj, grid, jacobian)

    return wrapper


def one_step(r, _, theta, fun, fun_dr):
    r = jnp.abs(r - fun(r, theta) / fun_dr(r, theta))
    return r, None


@partial(jit, static_argnums=(4,))
def step_r(r, theta, fun, fun_dr, N=20):
    one_step_partial = jax.tree_util.Partial(
        one_step, theta=theta, fun=fun, fun_dr=fun_dr
    )
    new_r = jax.lax.scan(one_step_partial, r, xs=jnp.arange(N))[0]
    return jnp.stack([new_r * jnp.sin(theta), new_r * jnp.cos(theta)]).T


class OperateDeflections:
    """
    Packages methods which manipulate the 2D deflection angle map returned from the `deflections_yx_2d_from` function
    of a mass object (e.g. a `MassProfile`, `Galaxy`).

    The majority of methods are those which from the 2D deflection angle map compute lensing quantities like a 2D
    shear field, magnification map or the Einstein Radius.

    The methods in `CalcLens` are passed to the mass object to provide a concise API.

    Parameters
    ----------
    deflections_yx_2d_from
        The function which returns the mass object's 2D deflection angles.
    """

    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        raise NotImplementedError

    def deflections_yx_scalar(self, y, x, pixel_scales):

        # A version of the deflection function that takes in two scalars
        # and outputs a 2D vector.  Needed for JAX auto differentiation.

        mask = aa.Mask2D.all_false(
            shape_native=(1, 1),
            pixel_scales=pixel_scales,
        )

        g = aa.Grid2D(
            values=jnp.stack((y.reshape(1), x.reshape(1)), axis=-1), mask=mask
        )

        return self.deflections_yx_2d_from(g).squeeze()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__ and self.__class__ is other.__class__

    def time_delay_geometry_term_from(self, grid) -> aa.Array2D:
        """
            Returns the geometric time delay term of the Fermat potential for a given grid of image-plane positions.

            This term is given by:

        .. math::
                \[\tau_{\text{geom}}(\boldsymbol{\theta}) = \frac{1}{2} |\boldsymbol{\theta} - \boldsymbol{\beta}|^2\]

            where:
            - \( \boldsymbol{\theta} \) is the image-plane coordinate,
            - \( \boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta}) \) is the source-plane coordinate,
            - \( \boldsymbol{\alpha} \) is the deflection angle at each image-plane coordinate.

            Parameters
            ----------
            grid
                The 2D grid of (y,x) arc-second coordinates the deflection angles and time delay geometric term are computed
                on.

            Returns
            -------
            The geometric time delay term at each grid position.
        """
        deflections = self.deflections_yx_2d_from(grid=grid)

        src_y = grid[:, 0] - deflections[:, 0]
        src_x = grid[:, 1] - deflections[:, 1]

        delay = 0.5 * ((grid[:, 0] - src_y) ** 2 + (grid[:, 1] - src_x) ** 2)

        if isinstance(grid, aa.Grid2DIrregular):
            return aa.ArrayIrregular(values=delay)
        return aa.Array2D(values=delay, mask=grid.mask)

    def fermat_potential_from(self, grid) -> aa.Array2D:
        """
        Returns the Fermat potential for a given grid of image-plane positions.

        This is the sum of the geometric time delay term and the gravitational (Shapiro) delay term (i.e. the lensing
        potential), and is given by:

        .. math::
            \[\phi(\boldsymbol{\theta}) = \frac{1}{2} |\boldsymbol{\theta} - \boldsymbol{\beta}|^2 - \psi(\boldsymbol{\theta})\]

        where:
        - \( \boldsymbol{\theta} \) is the image-plane coordinate,
        - \( \boldsymbol{\beta} = \boldsymbol{\theta} - \boldsymbol{\alpha}(\boldsymbol{\theta}) \) is the source-plane coordinate,
        - \( \psi(\boldsymbol{\theta}) \) is the lensing potential,
        - \( \phi(\boldsymbol{\theta}) \) is the Fermat potential.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the Fermat potential is computed on.

        Returns
        -------
        The Fermat potential at each grid position.
        """
        time_delay_geometry_term = self.time_delay_geometry_term_from(grid=grid)
        potential = self.potential_2d_from(grid=grid)

        fermat_potential = time_delay_geometry_term - potential

        if isinstance(grid, aa.Grid2DIrregular):
            return aa.ArrayIrregular(values=fermat_potential)
        return aa.Array2D(values=fermat_potential, mask=grid.mask)

    def time_delays_from(self, grid) -> aa.Array2D:
        """
        Returns the 2D time delay map of lensing object, which is computed as the deflection angles in the y and x
        directions multiplied by the y and x coordinates of the grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and time delay are computed on.
        """
        deflections_yx = self.deflections_yx_2d_from(grid=grid)

        return aa.Array2D(
            values=deflections_yx[:, 0] * grid[:, 0]
            + deflections_yx[:, 1] * grid[:, 1],
            mask=grid.mask,
        )

    def __hash__(self):
        return hash(repr(self))

    @precompute_jacobian
    def tangential_eigen_value_from(self, grid, jacobian=None) -> aa.Array2D:
        """
        Returns the tangential eigen values of lensing jacobian, which are given by the expression:

        `tangential_eigen_value = 1 - convergence - shear`

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and tangential eigen values are computed
            on.
        jacobian
            A precomputed lensing jacobian, which is passed throughout the `CalcLens` functions for efficiency.
        """
        convergence = self.convergence_2d_via_jacobian_from(
            grid=grid, jacobian=jacobian
        )

        shear_yx = self.shear_yx_2d_via_jacobian_from(grid=grid, jacobian=jacobian)

        return aa.Array2D(values=1 - convergence - shear_yx.magnitudes, mask=grid.mask)

    @precompute_jacobian
    def radial_eigen_value_from(self, grid, jacobian=None) -> aa.Array2D:
        """
        Returns the radial eigen values of lensing jacobian, which are given by the expression:

        radial_eigen_value = 1 - convergence + shear

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and radial eigen values are computed on.
        jacobian
            A precomputed lensing jacobian, which is passed throughout the `CalcLens` functions for efficiency.
        """
        convergence = self.convergence_2d_via_jacobian_from(
            grid=grid, jacobian=jacobian
        )

        shear = self.shear_yx_2d_via_jacobian_from(grid=grid, jacobian=jacobian)

        return aa.Array2D(values=1 - convergence + shear.magnitudes, mask=grid.mask)

    def magnification_2d_from(self, grid) -> aa.Array2D:
        """
        Returns the 2D magnification map of lensing object, which is computed as the inverse of the determinant of the
        jacobian.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and magnification map are computed on.
        """
        jacobian = self.jacobian_from(grid=grid)

        det_jacobian = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]

        return aa.Array2D(values=1 / det_jacobian, mask=grid.mask)

    def hessian_from(self, grid, buffer: float = 0.01, deflections_func=None) -> Tuple:
        """
        Returns the Hessian of the lensing object, where the Hessian is the second partial derivatives of the
        potential (see equation 55 https://inspirehep.net/literature/419263):

        `hessian_{i,j} = d^2 / dtheta_i dtheta_j`

        The Hessian is computed by evaluating the 2D deflection angles around every (y,x) coordinate on the input 2D
        grid map in four directions (positive y, negative y, positive x, negative x), exploiting how the deflection
        angles are the derivative of the potential.

        By using evaluating the deflection angles around each grid coordinate, the Hessian can therefore be computed
        using uniform or irregular 2D grids of (y,x). This can be slower, because x4 more deflection angle calculations
        are required, however it is more flexible in and therefore used throughout **PyAutoLens** by default.

        The Hessian is returned as a 4 entry tuple, which reflect its structure as a 2x2 matrix.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Hessian are computed on.
        buffer
            The spacing in the y and x directions around each grid coordinate where deflection angles are computed and
            used to estimate the derivative.
        """
        if deflections_func is None:
            deflections_func = self.deflections_yx_2d_from

        grid_shift_y_up = aa.Grid2DIrregular(
            values=jnp.stack([grid[:, 0] + buffer, grid[:, 1]], axis=1)
        )

        grid_shift_y_down = aa.Grid2DIrregular(
            values=jnp.stack([grid[:, 0] - buffer, grid[:, 1]], axis=1)
        )

        grid_shift_x_left = aa.Grid2DIrregular(
            values=jnp.stack([grid[:, 0], grid[:, 1] - buffer], axis=1)
        )

        grid_shift_x_right = aa.Grid2DIrregular(
            values=jnp.stack([grid[:, 0], grid[:, 1] + buffer], axis=1)
        )

        deflections_up = deflections_func(grid=grid_shift_y_up)
        deflections_down = deflections_func(grid=grid_shift_y_down)
        deflections_left = deflections_func(grid=grid_shift_x_left)
        deflections_right = deflections_func(grid=grid_shift_x_right)

        hessian_yy = 0.5 * (deflections_up[:, 0] - deflections_down[:, 0]) / buffer
        hessian_xy = 0.5 * (deflections_up[:, 1] - deflections_down[:, 1]) / buffer
        hessian_yx = 0.5 * (deflections_right[:, 0] - deflections_left[:, 0]) / buffer
        hessian_xx = 0.5 * (deflections_right[:, 1] - deflections_left[:, 1]) / buffer

        return hessian_yy, hessian_xy, hessian_yx, hessian_xx

    def convergence_2d_via_hessian_from(
        self, grid, buffer: float = 0.01
    ) -> aa.ArrayIrregular:
        """
        Returns the convergence of the lensing object, which is computed from the 2D deflection angle map via the
        Hessian using the expression (see equation 56 https://inspirehep.net/literature/419263):

        `convergence = 0.5 * (hessian_{0,0} + hessian_{1,1}) = 0.5 * (hessian_xx + hessian_yy)`

        By going via the Hessian, the convergence can be calculated at any (y,x) coordinate therefore using either a
        2D uniform or irregular grid.

        This calculation of the convergence is independent of analytic calculations defined within `MassProfile` objects
        and can therefore be used as a cross-check.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Hessian are computed on.
        buffer
            The spacing in the y and x directions around each grid coordinate where deflection angles are computed and
            used to estimate the derivative.
        """
        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, buffer=buffer
        )

        return aa.ArrayIrregular(values=0.5 * (hessian_yy + hessian_xx))

    def shear_yx_2d_via_hessian_from(
        self, grid, buffer: float = 0.01
    ) -> ShearYX2DIrregular:
        """
        Returns the 2D (y,x) shear vectors of the lensing object, which are computed from the 2D deflection angle map
        via the Hessian using the expressions (see equation 57 https://inspirehep.net/literature/419263):

        `shear_y = hessian_{1,0} =  hessian_{0,1} = hessian_yx = hessian_xy`
        `shear_x = 0.5 * (hessian_{0,0} - hessian_{1,1}) = 0.5 * (hessian_xx - hessian_yy)`

        By going via the Hessian, the shear vectors can be calculated at any (y,x) coordinate, therefore using either a
        2D uniform or irregular grid.

        This calculation of the shear vectors is independent of analytic calculations defined within `MassProfile`
        objects and can therefore be used as a cross-check.

        The result is returned as a `ShearYX2D` dats structure, which has shape [total_shear_vectors, 2], where
        entries for [:,0] are the gamma_2 values and entries for [:,1] are the gamma_1 values.

        Note therefore that this convention means the FIRST entries in the array are the gamma_2 values and the SECOND
        entries are the gamma_1 values.

        Parameters
        ----------
        grids
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Hessian are computed on.
        buffer
            The spacing in the y and x directions around each grid coordinate where deflection angles are computed and
            used to estimate the derivative.
        """

        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, buffer=buffer
        )

        gamma_1 = 0.5 * (hessian_xx - hessian_yy)
        gamma_2 = hessian_xy

        shear_yx_2d = jnp.zeros(shape=(grid.shape_slim, 2))

        shear_yx_2d[:, 0] = gamma_2
        shear_yx_2d[:, 1] = gamma_1

        return ShearYX2DIrregular(values=shear_yx_2d, grid=grid)

    def magnification_2d_via_hessian_from(
        self, grid, buffer: float = 0.01, deflections_func=None
    ) -> aa.ArrayIrregular:
        """
        Returns the 2D magnification map of lensing object, which is computed from the 2D deflection angle map
        via the Hessian using the expressions (see equation 60 https://inspirehep.net/literature/419263):

        `magnification = 1.0 / det(Jacobian) = 1.0 / abs((1.0 - convergence)**2.0 - shear**2.0)`
        `magnification = (1.0 - hessian_{0,0}) * (1.0 - hessian_{1, 1)) - hessian_{0,1}*hessian_{1,0}`
        `magnification = (1.0 - hessian_xx) * (1.0 - hessian_yy)) - hessian_xy*hessian_yx`

        By going via the Hessian, the magnification can be calculated at any (y,x) coordinate, therefore using either a
        2D uniform or irregular grid.

        This calculation of the magnification is independent of calculations using the Jacobian and can therefore be
        used as a cross-check.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and magnification map are computed on.
        """
        hessian_yy, hessian_xy, hessian_yx, hessian_xx = self.hessian_from(
            grid=grid, buffer=buffer, deflections_func=deflections_func
        )

        det_A = (1 - hessian_xx) * (1 - hessian_yy) - hessian_xy * hessian_yx

        return aa.ArrayIrregular(values=1.0 / det_A)

    def contour_list_from(self, grid, contour_array):
        grid_contour = aa.Grid2DContour(
            grid=grid,
            pixel_scales=grid.pixel_scales,
            shape_native=grid.shape_native,
            contour_array=contour_array.native,
        )

        return grid_contour.contour_list

    def tangential_critical_curve_list_from(
        self,
        grid,
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all tangential critical curves of the lensing system, which are computed as follows:

        1) Compute the tangential eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the tangential eigen values that are zero using a marching squares algorithm.

        Due to the use of a marching squares algorithm that requires the zero values of the tangential eigen values to
        be computed, critical curves can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and tangential eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            critical curve to be computed more accurately using a higher resolution grid.
        """
        tangential_eigen_values = self.tangential_eigen_value_from(grid=grid)

        return self.contour_list_from(grid=grid, contour_array=tangential_eigen_values)

    def radial_critical_curve_list_from(
        self,
        grid,
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all radial critical curves of the lensing system, which are computed as follows:

        1) Compute the radial eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the radial eigen values that are zero using a marching squares algorithm.

        Due to the use of a marching squares algorithm that requires the zero values of the radial eigen values to
        be computed, this critical curves can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and radial eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            critical curve to be computed more accurately using a higher resolution grid.
        """
        radial_eigen_values = self.radial_eigen_value_from(grid=grid)

        return self.contour_list_from(grid=grid, contour_array=radial_eigen_values)

    def tangential_caustic_list_from(
        self,
        grid,
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all tangential caustics of the lensing system, which are computed as follows:

        1) Compute the tangential eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the tangential eigen values that are zero using a marching squares algorithm.
        3) Compute the lensing system's deflection angles at the (y,x) coordinates of the tangential critical curve
           contours and ray-trace it to the source-plane, therefore forming the tangential caustics.

        Due to the use of a marching squares algorithm that requires the zero values of the tangential eigen values to
        be computed, caustics can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and tangential eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """

        tangential_critical_curve_list = self.tangential_critical_curve_list_from(
            grid=grid
        )

        tangential_caustic_list = []

        for tangential_critical_curve in tangential_critical_curve_list:
            deflections_critical_curve = self.deflections_yx_2d_from(
                grid=tangential_critical_curve
            )

            tangential_caustic_list.append(
                tangential_critical_curve - deflections_critical_curve
            )

        return tangential_caustic_list

    def radial_caustic_list_from(
        self,
        grid,
    ) -> List[aa.Grid2DIrregular]:
        """
        Returns all radial caustics of the lensing system, which are computed as follows:

        1) Compute the radial eigen values for every coordinate on the input grid via the Jacobian.
        2) Find contours of all values in the radial eigen values that are zero using a marching squares algorithm.
        3) Compute the lensing system's deflection angles at the (y,x) coordinates of the radial critical curve
           contours and ray-trace it to the source-plane, therefore forming the radial caustics.

        Due to the use of a marching squares algorithm that requires the zero values of the radial eigen values to
        be computed, this caustics can only be calculated using the Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and radial eigen values are computed
            on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """

        radial_critical_curve_list = self.radial_critical_curve_list_from(grid=grid)

        radial_caustic_list = []

        for radial_critical_curve in radial_critical_curve_list:
            deflections_critical_curve = self.deflections_yx_2d_from(
                grid=radial_critical_curve
            )

            radial_caustic_list.append(
                radial_critical_curve - deflections_critical_curve
            )

        return radial_caustic_list

    def radial_critical_curve_area_list_from(self, grid) -> List[float]:
        """
        Returns the surface area within each radial critical curve as a list, the calculation of which is described in
        the function `radial_critical_curve_list_from()`.

        The area is computed via a line integral.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.


        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the radial critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        radial_critical_curve_list = self.radial_critical_curve_list_from(grid=grid)

        return self.area_within_curve_list_from(curve_list=radial_critical_curve_list)

    def tangential_critical_curve_area_list_from(
        self,
        grid,
    ) -> List[float]:
        """
        Returns the surface area within each tangential critical curve as a list, the calculation of which is
        described in the function `tangential_critical_curve_list_from()`.

        The area is computed via a line integral.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        tangential_critical_curve_list = self.tangential_critical_curve_list_from(
            grid=grid
        )

        return self.area_within_curve_list_from(
            curve_list=tangential_critical_curve_list
        )

    def area_within_curve_list_from(
        self, curve_list: List[aa.Grid2DIrregular]
    ) -> List[float]:
        area_within_each_curve_list = []

        for curve in curve_list:
            x, y = curve[:, 0], curve[:, 1]
            area = jnp.abs(0.5 * jnp.sum(y[:-1] * jnp.diff(x) - x[:-1] * jnp.diff(y)))
            area_within_each_curve_list.append(area)

        return area_within_each_curve_list

    def einstein_radius_list_from(
        self,
        grid,
    ):
        """
        Returns a list of the Einstein radii corresponding to the area within each tangential critical curve.

        Each Einstein radius is defined as the radius of the circle which contains the same area as the area within
        each tangential critical curve.

        This definition is sometimes referred to as the "effective Einstein radius" in the literature and is commonly
        adopted in studies, for example the SLACS series of papers.

        The calculation of the tangential critical curves and their areas is described in the functions
         `tangential_critical_curve_list_from()` and `tangential_critical_curve_area_list_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        try:
            area_list = self.tangential_critical_curve_area_list_from(grid=grid)
            return [jnp.sqrt(area / jnp.pi) for area in area_list]
        except TypeError:
            raise TypeError("The grid input was unable to estimate the Einstein Radius")

    def einstein_radius_from(
        self,
        grid,
    ):
        """
        Returns the Einstein radius corresponding to the area within the tangential critical curve.

        The Einstein radius is defined as the radius of the circle which contains the same area as the area within
        the tangential critical curve.

        This definition is sometimes referred to as the "effective Einstein radius" in the literature and is commonly
        adopted in studies, for example the SLACS series of papers.

        If there are multiple tangential critical curves (e.g. because the mass distribution is complex) this function
        raises an error, and the function `einstein_radius_list_from()` should be used instead.

        The calculation of the tangential critical curves and their areas is described in the functions
         `tangential_critical_curve_list_from()` and `tangential_critical_curve_area_list_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential
            critical curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """

        einstein_radii_list = self.einstein_radius_list_from(grid=grid)

        if len(einstein_radii_list) > 1:
            logger.info(
                """
                There are multiple tangential critical curves, and the computed Einstein radius is the sum of 
                all of them. Check the `einstein_radius_list_from` function for the individual Einstein. 
            """
            )

        return sum(einstein_radii_list)

    def einstein_mass_angular_list_from(
        self,
        grid,
    ) -> List[float]:
        """
        Returns a list of the angular Einstein massses corresponding to the area within each tangential critical curve.

        The angular Einstein mass is defined as: `einstein_mass = pi * einstein_radius ** 2.0` where the Einstein
        radius is the radius of the circle which contains the same area as the area within the tangential critical
        curve.

        The Einstein mass is returned in units of arcsecond**2.0 and requires division by the lensing critical surface
        density \sigma_cr to be converted to physical units like solar masses (see `autogalaxy.util.cosmology_util`).

        This definition of Eisntein radius (and therefore mass) is sometimes referred to as the "effective Einstein
        radius" in the literature and is commonly adopted in studies, for example the SLACS series of papers.

        The calculation of the einstein radius is described in the function `einstein_radius_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        einstein_radius_list = self.einstein_radius_list_from(grid=grid)
        return [jnp.pi * einstein_radius**2 for einstein_radius in einstein_radius_list]

    def einstein_mass_angular_from(
        self,
        grid,
    ) -> float:
        """
        Returns the Einstein radius corresponding to the area within the tangential critical curve.

        The angular Einstein mass is defined as: `einstein_mass = pi * einstein_radius ** 2.0` where the Einstein
        radius is the radius of the circle which contains the same area as the area within the tangential critical
        curve.

        The Einstein mass is returned in units of arcsecond**2.0 and requires division by the lensing critical surface
        density \sigma_cr to be converted to physical units like solar masses (see `autogalaxy.util.cosmology_util`).

        This definition of Eisntein radius (and therefore mass) is sometimes referred to as the "effective Einstein
        radius" in the literature and is commonly adopted in studies, for example the SLACS series of papers.

        The calculation of the einstein radius is described in the function `einstein_radius_from()`.

        Due to the use of a marching squares algorithm to estimate the critical curve, this function can only use the
        Jacobian and a uniform 2D grid.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles used to calculate the tangential critical
            curve are computed on.
        pixel_scale
            If input, the `evaluation_grid` decorator creates the 2D grid at this resolution, therefore enabling the
            caustic to be computed more accurately using a higher resolution grid.
        """
        einstein_mass_angular_list = self.einstein_mass_angular_list_from(grid=grid)

        if len(einstein_mass_angular_list) > 1:
            logger.info(
                """
                There are multiple tangential critical curves, and the computed Einstein mass is the sum of 
                all of them. Check the `einstein_mass_list_from` function for the individual Einstein. 
            """
            )

        return einstein_mass_angular_list[0]

    def jacobian_stack(self, y, x, pixel_scales):
        return jnp.stack(
            jax.jacfwd(self.deflections_yx_scalar, argnums=(0, 1))(y, x, pixel_scales)
        )

    def jacobian_stack_vector(self, y, x, pixel_scales):
        return jnp.vectorize(
            jax.tree_util.Partial(self.jacobian_stack, pixel_scales=pixel_scales),
            signature="(),()->(i,i)",
        )(y, x)

    def convergence_mag_shear_yx(self, y, x):
        J = self.jacobian_stack_vector(y, x, 0.05)
        K = 0.5 * (J[..., 0, 0] + J[..., 1, 1])
        mag_shear = 0.5 * jnp.sqrt(
            (J[..., 0, 1] + J[..., 1, 0]) ** 2 + (J[..., 0, 0] - J[..., 1, 1]) ** 2
        )
        return K, mag_shear

    @partial(jit, static_argnums=(0,))
    def tangential_eigen_value_yx(self, y, x):
        K, mag_shear = self.convergence_mag_shear_yx(y, x)
        return 1 - K - mag_shear

    @partial(jit, static_argnums=(0, 3))
    def tangential_eigen_value_rt(self, r, theta, centre=(0.0, 0.0)):
        y = r * jnp.sin(theta) + centre[0]
        x = r * jnp.cos(theta) + centre[1]
        return self.tangential_eigen_value_yx(y, x)

    @partial(jit, static_argnums=(0, 3))
    def grad_r_tangential_eigen_value(self, r, theta, centre=(0.0, 0.0)):
        # ignore `self` with the `argnums` below
        tangential_eigen_part = partial(self.tangential_eigen_value_rt, centre=centre)
        return jnp.vectorize(
            jax.jacfwd(tangential_eigen_part, argnums=(0,)), signature="(),()->()"
        )(r, theta)[0]

    @partial(jit, static_argnums=(0,))
    def radial_eigen_value_yx(self, y, x):
        K, mag_shear = self.convergence_mag_shear_yx(y, x)
        return 1 - K + mag_shear

    @partial(jit, static_argnums=(0, 3))
    def radial_eigen_value_rt(self, r, theta, centre=(0.0, 0.0)):
        y = r * jnp.sin(theta) + centre[0]
        x = r * jnp.cos(theta) + centre[1]
        return self.radial_eigen_value_yx(y, x)

    @partial(jit, static_argnums=(0, 3))
    def grad_r_radial_eigen_value(self, r, theta, centre=(0.0, 0.0)):
        # ignore `self` with the `argnums` below
        radial_eigen_part = partial(self.radial_eigen_value_rt, centre=centre)
        return jnp.vectorize(
            jax.jacfwd(radial_eigen_part, argnums=(0,)), signature="(),()->()"
        )(r, theta)[0]

    def tangential_critical_curve_jax(
        self,
        init_r=0.1,
        init_centre=(0.0, 0.0),
        n_points=300,
        n_steps=20,
        threshold=1e-5,
    ):
        """
        Returns all tangential critical curves of the lensing system, which are computed as follows:

        1) Create a set of `n_points` initial points in a circle of radius `init_r` and centred on `init_centre`
        2) Apply `n_steps` of Newton's method to these points in the "radial" direction only (i.e. keeping angle fixed).
        Jax's auto differentiation is used to find the radial derivatives of the tangential eigen value function for
        this step.
        3) Filter the results and only keep point that have their tangential eigen value `threshold` of 0

        No underlying grid is needed for the method, but the quality of the results are dependent on the initial
        circle of points.

        Parameters
        ----------
        init_r : float
            Radius of the circle of initial guess points
        init_centre : tuple
            centre of the circle of initial guess points as `(y, x)`
        n_points : Int
            Number of initial guess points to create (evenly spaced in angle around `init_centre`)
        n_steps : Int
            Number of iterations of Newton's method to apply
        threshold : float
            Only keep points whose tangential eigen value is within this value of zero (inclusive)
        """
        r = jnp.ones(n_points) * init_r
        theta = jnp.linspace(0, 2 * jnp.pi, n_points + 1)[:-1]
        new_yx = step_r(
            r,
            theta,
            jax.tree_util.Partial(self.tangential_eigen_value_rt, centre=init_centre),
            jax.tree_util.Partial(
                self.grad_r_tangential_eigen_value, centre=init_centre
            ),
            n_steps,
        )
        new_yx = new_yx + jnp.array(init_centre)
        # filter out nan values
        fdx = jnp.isfinite(new_yx).all(axis=1)
        new_yx = new_yx[fdx]
        # filter out failed points
        value = jnp.abs(self.tangential_eigen_value_yx(new_yx[:, 0], new_yx[:, 1]))
        gdx = value <= threshold
        return aa.structures.grids.irregular_2d.Grid2DIrregular(values=new_yx[gdx])

    def radial_critical_curve_jax(
        self,
        init_r=0.01,
        init_centre=(0.0, 0.0),
        n_points=300,
        n_steps=20,
        threshold=1e-5,
    ):
        """
        Returns all radial critical curves of the lensing system, which are computed as follows:

        1) Create a set of `n_points` initial points in a circle of radius `init_r` and centred on `init_centre`
        2) Apply `n_steps` of Newton's method to these points in the "radial" direction only (i.e. keeping angle fixed).
        Jax's auto differentiation is used to find the radial derivatives of the radial eigen value function for
        this step.
        3) Filter the results and only keep point that have their radial eigen value `threshold` of 0

        No underlying grid is needed for the method, but the quality of the results are dependent on the initial
        circle of points.

        Parameters
        ----------
        init_r : float
            Radius of the circle of initial guess points
        init_centre : tuple
            centre of the circle of initial guess points as `(y, x)`
        n_points : Int
            Number of initial guess points to create (evenly spaced in angle around `init_centre`)
        n_steps : Int
            Number of iterations of Newton's method to apply
        threshold : float
            Only keep points whose radial eigen value is within this value of zero (inclusive)
        """
        r = jnp.ones(n_points) * init_r
        theta = jnp.linspace(0, 2 * jnp.pi, n_points + 1)[:-1]
        new_yx = step_r(
            r,
            theta,
            jax.tree_util.Partial(self.radial_eigen_value_rt, centre=init_centre),
            jax.tree_util.Partial(self.grad_r_radial_eigen_value, centre=init_centre),
            n_steps,
        )
        new_yx = new_yx + jnp.array(init_centre)
        # filter out nan values
        fdx = jnp.isfinite(new_yx).all(axis=1)
        new_yx = new_yx[fdx]
        # filter out failed points
        value = jnp.abs(self.radial_eigen_value_yx(new_yx[:, 0], new_yx[:, 1]))
        gdx = value <= threshold
        return aa.structures.grids.irregular_2d.Grid2DIrregular(values=new_yx[gdx])

    def jacobian_from(self, grid):
        """
        Returns the Jacobian of the lensing object, which is computed by taking the gradient of the 2D deflection
        angle map in four direction (positive y, negative y, positive x, negative x).

        By using the `np.gradient` method the Jacobian can therefore only be computed using uniform 2D grids of (y,x)
        coordinates, and does not support irregular grids. For this reason, calculations by default use the Hessian,
        which is slower to compute because more deflection angle calculations are necessary but more flexible in
        general.

        The Jacobian is returned as a list of lists, which reflect its structure as a 2x2 matrix.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Jacobian are computed on.
        """
        A = self.jacobian_stack_vector(
            grid.array[:, 0], grid.array[:, 1], grid.pixel_scales
        )
        a = jnp.eye(2).reshape(1, 2, 2) - A
        return [
            [
                aa.Array2D(values=a[..., 1, 1], mask=grid.mask),
                aa.Array2D(values=a[..., 1, 0], mask=grid.mask),
            ],
            [
                aa.Array2D(values=a[..., 0, 1], mask=grid.mask),
                aa.Array2D(values=a[..., 0, 0], mask=grid.mask),
            ],
        ]

        # transpose the result
        # use `moveaxis` as grid might not be nx2
        # return jnp.moveaxis(jnp.moveaxis(a, -1, 0), -1, 0)

    @precompute_jacobian
    def convergence_2d_via_jacobian_from(self, grid, jacobian=None) -> aa.Array2D:
        """
        Returns the convergence of the lensing object, which is computed from the 2D deflection angle map via the
        Jacobian using the expression (see equation 58 https://inspirehep.net/literature/419263):

        `convergence = 1.0 - 0.5 * (jacobian_{0,0} + jacobian_{1,1}) = 0.5 * (jacobian_xx + jacobian_yy)`

        By going via the Jacobian, the convergence must be calculated using 2D uniform grid.

        This calculation of the convergence is independent of analytic calculations defined within `MassProfile`
        objects and the calculation via the Hessian. It can therefore be used as a cross-check.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Jacobian are computed on.
        jacobian
            A precomputed lensing jacobian, which is passed throughout the `CalcLens` functions for efficiency.
        """
        convergence = 1 - 0.5 * (jacobian[0][0] + jacobian[1][1])

        return aa.Array2D(values=convergence, mask=grid.mask)

    @precompute_jacobian
    def shear_yx_2d_via_jacobian_from(
        self, grid, jacobian=None
    ) -> Union[ShearYX2D, ShearYX2DIrregular]:
        """
        Returns the 2D (y,x) shear vectors of the lensing object, which are computed from the 2D deflection angle map
        via the Jacobian using the expression (see equation 58 https://inspirehep.net/literature/419263):

        `shear_y = -0.5 * (jacobian_{0,1} + jacobian_{1,0} = -0.5 * (jacobian_yx + jacobian_xy)`
        `shear_x = 0.5 * (jacobian_{1,1} + jacobian_{0,0} = 0.5 * (jacobian_yy + jacobian_xx)`

        By going via the Jacobian, the convergence must be calculated using 2D uniform grid.

        This calculation of the shear vectors is independent of analytic calculations defined within `MassProfile`
        objects and the calculation via the Hessian. It can therefore be used as a cross-check.

        Parameters
        ----------
        grid
            The 2D grid of (y,x) arc-second coordinates the deflection angles and Jacobian are computed on.
        jacobian
            A precomputed lensing jacobian, which is passed throughout the `CalcLens` functions for efficiency.
        """
        shear_y = -0.5 * (jacobian[0][1] + jacobian[1][0]).array
        shear_x = 0.5 * (jacobian[1][1] - jacobian[0][0]).array
        shear_yx_2d = jnp.stack([shear_y, shear_x]).T

        if isinstance(grid, aa.Grid2DIrregular):
            return ShearYX2DIrregular(values=shear_yx_2d, grid=grid)
        return ShearYX2D(values=shear_yx_2d, grid=grid, mask=grid.mask)
