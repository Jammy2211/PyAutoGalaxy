import autofit as af
import numpy as np
from autoarray.structures import grids
from autogalaxy import dimensions as dim
from autogalaxy.util import cosmology_util, convert
import typing


class GeometryProfile(dim.DimensionsProfile):
    @af.map_types
    def __init__(self, centre: dim.Position = (0.0, 0.0)):
        """An abstract geometry profile, which describes profiles with y and x centre Cartesian coordinates
        
        Parameters
        -----------
        centre : (float, float)
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


class SphericalProfile(GeometryProfile):
    @af.map_types
    def __init__(self, centre: dim.Position = (0.0, 0.0)):
        """ A spherical profile, which describes profiles with y and x centre Cartesian coordinates.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        """
        super(SphericalProfile, self).__init__(centre=centre)

    @grids.grid_like_to_structure
    @grids.transform
    def grid_to_grid_radii(self, grid):
        """Convert a grid of (y, x) coordinates to a grid of their circular radii.

        If the coordinates have not been transformed to the profile's centre, this is performed automatically.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the reference frame of the profile.
        """
        return np.sqrt(np.add(np.square(grid[:, 0]), np.square(grid[:, 1])))

    def grid_angle_to_profile(self, grid_thetas):
        """The angle between each (y,x) coordinate on the grid and the profile, in radians.
        
        Parameters
        -----------
        grid_thetas : ndarray
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        return np.cos(grid_thetas), np.sin(grid_thetas)

    @grids.grid_like_to_structure
    def grid_to_grid_cartesian(self, grid, radius):
        """
        Convert a grid of (y,x) coordinates with their specified circular radii to their original (y,x) Cartesian 
        coordinates.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the reference frame of the profile.
        radius : ndarray
            The circular radius of each coordinate from the profile center.
        """
        grid_thetas = np.arctan2(grid[:, 0], grid[:, 1])
        cos_theta, sin_theta = self.grid_angle_to_profile(grid_thetas=grid_thetas)
        return np.multiply(radius[:, None], np.vstack((sin_theta, cos_theta)).T)

    @grids.grid_like_to_structure
    def transform_grid_to_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates to the reference frame of the profile, including a translation to \
        its centre.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        """
        transformed = np.subtract(grid, self.centre)
        return grids.GridTransformedNumpy(grid=transformed)

    @grids.grid_like_to_structure
    def transform_grid_from_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates from the reference frame of the profile to the original observer \
        reference frame, including a translation from the profile's centre.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the reference frame of the profile.
        """
        transformed = np.add(grid, self.centre)
        return transformed.view(grids.GridTransformedNumpy)


class EllipticalProfile(SphericalProfile):
    @af.map_types
    def __init__(
        self,
        centre: dim.Position = (0.0, 0.0),
        elliptical_comps: typing.Tuple[float, float] = (0.0, 0.0),
    ):
        """ An elliptical profile, which describes profiles with y and x centre Cartesian coordinates, an axis-ratio \
        and rotational angle phi.

        Parameters
        ----------
        centre: (float, float)
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps : (float, float)
            The first and second ellipticity components of the elliptical coordinate system, where
            fac = (1 - axis_ratio) / (1 + axis_ratio), ellip_y = fac * sin(2*phi) and ellip_x = fac * cos(2*phi).

        Attributes
        ----------
        axis_ratio : float
            Ratio of light profiles ellipse's minor and major axes (b/a).
        phi : float
            Rotation angle of light profile counter-clockwise from positive x-axis.
        """
        super(EllipticalProfile, self).__init__(centre=centre)

        self.elliptical_comps = elliptical_comps

        axis_ratio, phi = convert.axis_ratio_and_phi_from(
            elliptical_comps=elliptical_comps
        )

        self.axis_ratio = axis_ratio
        self.phi = phi

    @classmethod
    def from_axis_ratio_and_phi(
        cls,
        centre: dim.Position = (0.0, 0.0),
        axis_ratio: float = 1.0,
        phi: float = 0.0,
    ):

        elliptical_comps = convert.elliptical_comps_from(axis_ratio=axis_ratio, phi=phi)
        return cls(centre=centre, elliptical_comps=elliptical_comps)

    @property
    def phi_radians(self):
        return np.radians(self.phi)

    @property
    def cos_phi(self):
        return self.cos_and_sin_from_x_axis()[0]

    @property
    def sin_phi(self):
        return self.cos_and_sin_from_x_axis()[1]

    def cos_and_sin_from_x_axis(self):
        """ Determine the sin and cosine of the angle between the profile's ellipse and the positive x-axis, \
        counter-clockwise. """
        phi_radians = np.radians(self.phi)
        return np.cos(phi_radians), np.sin(phi_radians)

    def grid_angle_to_profile(self, grid_thetas):
        """The angle between each angle theta on the grid and the profile, in radians.

        Parameters
        -----------
        grid_thetas : ndarray
            The angle theta counter-clockwise from the positive x-axis to each coordinate in radians.
        """
        theta_coordinate_to_profile = np.add(grid_thetas, -self.phi_radians)
        return np.cos(theta_coordinate_to_profile), np.sin(theta_coordinate_to_profile)

    @grids.grid_like_to_structure
    def rotate_grid_from_profile(self, grid_elliptical):
        """ Rotate a grid of elliptical (y,x) coordinates from the reference frame of the profile back to the \
        unrotated coordinate grid reference frame (coordinates are not shifted back to their original centre).

        This routine is used after computing deflection angles in the reference frame of the profile, so that the \
        deflection angles can be re-rotated to the frame of the original coordinates before performing ray-tracing.

        Parameters
        ----------
        grid_elliptical : grid_like
            The (y, x) coordinates in the reference frame of an elliptical profile.
        """
        y = np.add(
            np.multiply(grid_elliptical[:, 1], self.sin_phi),
            np.multiply(grid_elliptical[:, 0], self.cos_phi),
        )
        x = np.add(
            np.multiply(grid_elliptical[:, 1], self.cos_phi),
            -np.multiply(grid_elliptical[:, 0], self.sin_phi),
        )
        return np.vstack((y, x)).T

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def grid_to_elliptical_radii(self, grid):
        """ Convert a grid of (y,x) coordinates to an elliptical radius.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the reference frame of the elliptical profile.
        """
        return np.sqrt(
            np.add(
                np.square(grid[:, 1]), np.square(np.divide(grid[:, 0], self.axis_ratio))
            )
        )

    @grids.grid_like_to_structure
    @grids.transform
    @grids.relocate_to_radial_minimum
    def grid_to_eccentric_radii(self, grid):
        """Convert a grid of (y,x) coordinates to an eccentric radius, which is (1.0/axis_ratio) * elliptical radius \
        and used to define light profile half-light radii using circular radii.

        If the coordinates have not been transformed to the profile's geometry, this is performed automatically.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the reference frame of the elliptical profile.
        """
        return np.multiply(
            np.sqrt(self.axis_ratio), self.grid_to_elliptical_radii(grid)
        ).view(np.ndarray)

    @grids.grid_like_to_structure
    def transform_grid_to_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates to the reference frame of the profile, including a translation to \
        its centre and a rotation to it orientation.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the original reference frame of the grid.
        """
        if self.__class__.__name__.startswith("Spherical"):
            return super().transform_grid_to_reference_frame(
                grid=grids.GridTransformedNumpy(grid=grid)
            )
        shifted_coordinates = np.subtract(grid, self.centre)
        radius = np.sqrt(np.sum(shifted_coordinates ** 2.0, 1))
        theta_coordinate_to_profile = (
            np.arctan2(shifted_coordinates[:, 0], shifted_coordinates[:, 1])
            - self.phi_radians
        )
        transformed = np.vstack(
            (
                radius * np.sin(theta_coordinate_to_profile),
                radius * np.cos(theta_coordinate_to_profile),
            )
        ).T
        return grids.GridTransformedNumpy(grid=transformed)

    @grids.grid_like_to_structure
    def transform_grid_from_reference_frame(self, grid):
        """Transform a grid of (y,x) coordinates from the reference frame of the profile to the original observer \
        reference frame, including a rotation to its original orientation and a translation from the profile's centre.

        Parameters
        ----------
        grid : grid_like
            The (y, x) coordinates in the reference frame of the profile.
        """
        if self.__class__.__name__.startswith("Spherical"):
            return super().transform_grid_from_reference_frame(
                grid=grids.GridTransformedNumpy(grid=grid)
            )

        y = np.add(
            np.add(
                np.multiply(grid[:, 1], self.sin_phi),
                np.multiply(grid[:, 0], self.cos_phi),
            ),
            self.centre[0],
        )
        x = np.add(
            np.add(
                np.multiply(grid[:, 1], self.cos_phi),
                -np.multiply(grid[:, 0], self.sin_phi),
            ),
            self.centre[1],
        )
        return np.vstack((y, x)).T

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
