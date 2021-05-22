import numpy as np
from autoarray.geometry import geometry_util
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids import grid_decorators
from autogalaxy import convert

from typing import Tuple


class GeometryProfile:
    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0)):
        """An abstract geometry profile, which describes profiles with y and x centre Cartesian coordinates

        Parameters
        -----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        """

        self.centre = centre

    def transform_grid_to_reference_frame(self, grid):
        raise NotImplemented()

    def transform_grid_from_reference_frame(self, grid):
        raise NotImplemented()

    def __repr__(self):
        return "{}\n{}".format(
            self.__class__.__name__,
            "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()]),
        )

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class SphProfile(GeometryProfile):
    def __init__(self, centre: Tuple[float, float] = (0.0, 0.0)):
        """A spherical profile, which describes profiles with y and x centre Cartesian coordinates.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        """
        super().__init__(centre=centre)

    @grid_decorators.grid_2d_to_structure
    @grid_decorators.transform
    def grid_to_grid_radii(self, grid):
        """Convert a grid of (y, x) coordinates to a grid of their circular radii.

        If the coordinates have not been transformed to the profile's centre, this is performed automatically.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the profile.
        """
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    def grid_angle_to_profile(self, grid_thetas):
        """The angle between each (y,x) coordinate on the grid and the profile, in radians.

        Parameters
        -----------
        grid_thetas : np.ndarray
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        return np.cos(grid_thetas), np.sin(grid_thetas)

    @grid_decorators.grid_2d_to_structure
    def grid_to_grid_cartesian(self, grid, radius):
        """
        Convert a grid of (y,x) coordinates with their specified circular radii to their original (y,x) Cartesian
        coordinates.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the profile.
        radius : np.ndarray
            The circular radius of each coordinate from the profile center.
        """
        grid_thetas = np.arctan2(grid[:, 0], grid[:, 1])
        cos_theta, sin_theta = self.grid_angle_to_profile(grid_thetas=grid_thetas)
        return np.multiply(radius[:, None], np.vstack((sin_theta, cos_theta)).T)

    @grid_decorators.grid_2d_to_structure
    def transform_grid_to_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates to the reference frame of the profile, including a translation to \
        its centre.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        """
        transformed = np.subtract(grid, self.centre)
        return grid_2d.Grid2DTransformedNumpy(grid=transformed)

    @grid_decorators.grid_2d_to_structure
    def transform_grid_from_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates from the reference frame of the profile to the original observer \
        reference frame, including a translation from the profile's centre.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the profile.
        """
        transformed = np.add(grid, self.centre)
        return transformed.view(grid_2d.Grid2DTransformedNumpy)


class EllProfile(SphProfile):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
    ):
        """ An elliptical profile, which describes profiles with y and x centre Cartesian coordinates, an axis-ratio \
        and rotational angle.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*angle) and ellip_x = fac * cos(2*angle).

        Attributes
        ----------
        axis_ratio
            Ratio of light profiles ellipse's minor and major axes (b/a).
        angle
            Rotation angle of light profile counter-clockwise from positive x-axis.
        """
        super().__init__(centre=centre)

        self.elliptical_comps = elliptical_comps

        axis_ratio, angle = convert.axis_ratio_and_phi_from(
            elliptical_comps=elliptical_comps
        )

        self.axis_ratio = axis_ratio
        self.angle = angle

    @classmethod
    def from_axis_ratio_and_phi(
        cls,
        centre: Tuple[float, float] = (0.0, 0.0),
        axis_ratio: float = 1.0,
        angle: float = 0.0,
    ):

        elliptical_comps = convert.elliptical_comps_from(
            axis_ratio=axis_ratio, angle=angle
        )
        return cls(centre=centre, elliptical_comps=elliptical_comps)

    @property
    def phi_radians(self):
        return np.radians(self.angle)

    @property
    def cos_phi(self):
        return self.cos_and_sin_from_x_axis()[0]

    @property
    def sin_phi(self):
        return self.cos_and_sin_from_x_axis()[1]

    def cos_and_sin_from_x_axis(self):
        """ Determine the sin and cosine of the angle between the profile's ellipse and the positive x-axis, \
        counter-clockwise. """
        phi_radians = np.radians(self.angle)
        return np.cos(phi_radians), np.sin(phi_radians)

    def grid_angle_to_profile(self, grid_thetas):
        """The angle between each angle theta on the grid and the profile, in radians.

        Parameters
        -----------
        grid_thetas : np.ndarray
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        theta_coordinate_to_profile = np.add(grid_thetas, -self.phi_radians)
        return np.cos(theta_coordinate_to_profile), np.sin(theta_coordinate_to_profile)

    @grid_decorators.grid_2d_to_structure
    def rotate_grid_from_reference_frame(self, grid):
        """
        Rotate a grid of (y,x) coordinates which have been transformed to the elliptical reference frame of a profile
        back to the original unrotated coordinate grid reference frame.

        Note that unlike the method `transform_grid_from_reference_frame` the the coordinates are not
        translated back to the profile's original centre.

        This routine is used after computing deflection angles in the reference frame of the profile, so that the
        deflection angles can be re-rotated to the frame of the original coordinates before performing ray-tracing.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of an elliptical profile.
        """
        return geometry_util.transform_grid_2d_from_reference_frame(
            grid_2d=grid, centre=(0.0, 0.0), angle=self.angle
        )

    @grid_decorators.grid_2d_to_structure
    @grid_decorators.transform
    @grid_decorators.relocate_to_radial_minimum
    def grid_to_elliptical_radii(self, grid):
        """
        Convert a grid of (y,x) coordinates to an elliptical radius.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the elliptical profile.
        """
        return np.sqrt(
            np.add(
                np.square(grid[:, 1]), np.square(np.divide(grid[:, 0], self.axis_ratio))
            )
        )

    @grid_decorators.grid_2d_to_structure
    @grid_decorators.transform
    @grid_decorators.relocate_to_radial_minimum
    def grid_to_eccentric_radii(self, grid):
        """
        Convert a grid of (y,x) coordinates to an eccentric radius, which is (1.0/axis_ratio) * elliptical radius \
        and used to define light profile half-light radii using circular radii.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the elliptical profile.
        """
        return np.multiply(
            np.sqrt(self.axis_ratio), self.grid_to_elliptical_radii(grid)
        ).view(np.ndarray)

    @grid_decorators.grid_2d_to_structure
    def transform_grid_to_reference_frame(self, grid):
        """
        Transform a grid of (y,x) coordinates to the reference frame of the profile, including a translation to \
        its centre and a rotation to it orientation.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.__class__.__name__.startswith("Sph"):
            return super().transform_grid_to_reference_frame(
                grid=grid_2d.Grid2DTransformedNumpy(grid=grid)
            )
        transformed = geometry_util.transform_grid_2d_to_reference_frame(
            grid_2d=grid, centre=self.centre, angle=self.angle
        )
        return grid_2d.Grid2DTransformedNumpy(grid=transformed)

    @grid_decorators.grid_2d_to_structure
    def transform_grid_from_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates from the reference frame of the profile to the original observer \
        reference frame, including a rotation to its original orientation and a translation from the profile's centre.

        Parameters
        ----------
        grid
            The (y, x) coordinates in the reference frame of the profile.
        """
        if self.__class__.__name__.startswith("Sph"):
            return super().transform_grid_from_reference_frame(
                grid=grid_2d.Grid2DTransformedNumpy(grid=grid)
            )

        return geometry_util.transform_grid_2d_from_reference_frame(
            grid_2d=grid, centre=self.centre, angle=self.angle
        )

    def eta_u(self, u, coordinates):
        return np.sqrt(
            (
                u
                * (
                    (coordinates[1] ** 2)
                    + (coordinates[0] ** 2 / (1 - (1 - self.axis_ratio ** 2) * u))
                )
            )
        )
