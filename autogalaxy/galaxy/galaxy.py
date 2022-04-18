from itertools import count
from typing import Dict, List, Optional, Union

import numpy as np

import autoarray as aa
import autofit as af
from autoarray.inversion.pixelizations.abstract import AbstractPixelization
from autoarray.inversion.regularization.abstract import AbstractRegularization
from autoconf.dictable import Dictable
from autogalaxy import exc
from autogalaxy.profiles.geometry_profiles import GeometryProfile
from autogalaxy.profiles.light_profiles.light_profiles import LightProfile
from autogalaxy.profiles.mass_profiles import MassProfile
from autogalaxy.profiles.point_sources import Point
from autogalaxy.operate.image import OperateImageList
from autogalaxy.operate.deflections import OperateDeflections


def is_point_source(obj) -> bool:
    return isinstance(obj, Point)


def is_light_profile(obj) -> bool:
    return isinstance(obj, LightProfile)


def is_mass_profile(obj) -> bool:
    return isinstance(obj, MassProfile)


class Galaxy(af.ModelObject, OperateImageList, OperateDeflections, Dictable):
    """
    @DynamicAttrs
    """

    def __init__(
        self,
        redshift: float,
        pixelization: Optional[AbstractPixelization] = None,
        regularization: Optional[AbstractRegularization] = None,
        hyper_galaxy: Optional["HyperGalaxy"] = None,
        **kwargs,
    ):
        """
        Class representing a galaxy, which is composed of attributes used for fitting hyper_galaxies (e.g. light profiles, \
        mass profiles, pixelizations, etc.).
        
        All *has_* methods retun `True` if galaxy has that attribute, `False` if not.

        Parameters
        ----------
        redshift
            The redshift of the galaxy.
        pixelization : inversion.Pixelization
            The pixelization of the galaxy used to reconstruct an observed image using an inversion.
        regularization : inversion.Regularization
            The regularization of the pixel-grid used to reconstruct an observed using an inversion.
        hyper_galaxy
            The hyper_galaxies-parameters of the hyper_galaxies-galaxy, which is used for performing a hyper_galaxies-analysis on the noise-map.
            
        Attributes
        ----------
        light_profile_list
            A list of the galaxy's light profiles.
        mass_profile_list
            A list of the galaxy's mass profiles.
        hyper_model_image
            The best-fit model image to the observed image from a previous analysis
            search. This provides the total light attributed to each image pixel by the
            model.
        hyper_galaxy_image
            A model image of the galaxy (from light profiles or an inversion) from a
            previous analysis search.
        """
        super().__init__()
        self.redshift = redshift

        self.hyper_model_image = None
        self.hyper_galaxy_image = None

        for name, val in kwargs.items():

            if isinstance(val, list):
                raise exc.GalaxyException(
                    "One or more of the input light / mass profiles has been passed to the Galaxy object"
                    "as a list."
                    ""
                    "The Galaxy object cannot accept a list of light / mass profiles. "
                    ""
                    "Instead, pass these objects as a dictionary, where the key of each dictionary entry is"
                    "the name of the profile and the value is the profile, e.g.:"
                    ""
                    "{bulge : al.lp.EllSersic()}"
                    ""
                )

            setattr(self, name, val)

        self.pixelization = pixelization
        self.regularization = regularization

        if pixelization is not None and regularization is None:
            raise exc.GalaxyException(
                "If the galaxy has a pixelization, it must also have a regularization."
            )
        if pixelization is None and regularization is not None:
            raise exc.GalaxyException(
                "If the galaxy has a regularization, it must also have a pixelization."
            )

        self.hyper_galaxy = hyper_galaxy

    def __hash__(self):
        return self.id

    def __repr__(self):
        string = "Redshift: {}".format(self.redshift)
        if self.pixelization:
            string += "\nPixelization:\n{}".format(str(self.pixelization))
        if self.regularization:
            string += "\nRegularization:\n{}".format(str(self.regularization))
        if self.hyper_galaxy:
            string += "\nHyper Galaxy:\n{}".format(str(self.hyper_galaxy))
        if self.light_profile_list:
            string += "\nLight Profiles:\n{}".format(
                "\n".join(map(str, self.light_profile_list))
            )
        if self.mass_profile_list:
            string += "\nMass Profiles:\n{}".format(
                "\n".join(map(str, self.mass_profile_list))
            )
        return string

    def __eq__(self, other):
        return all(
            (
                isinstance(other, Galaxy),
                self.pixelization == other.pixelization,
                self.redshift == other.redshift,
                self.hyper_galaxy == other.hyper_galaxy,
                self.light_profile_list == other.light_profile_list,
                self.mass_profile_list == other.mass_profile_list,
            )
        )

    def dict(self) -> Dict:
        return {
            **{name: profile.dict() for name, profile in self.profile_dict.items()},
            **Dictable.dict(self),
        }

    @property
    def has_profile(self) -> bool:
        return len(self.mass_profile_list) + len(self.light_profile_list) > 0

    @property
    def light_profile_list(self) -> List[LightProfile]:
        return [value for value in self.__dict__.values() if is_light_profile(value)]

    @property
    def has_light_profile(self) -> bool:
        return len(self.light_profile_list) > 0

    def radial_projected_shape_slim_from(self, grid: aa.type.Grid1D2DLike) -> int:
        """
        To make 1D plots (e.g. `image_1d_from()`) from an input 2D grid, one uses that 2D grid to radially project
        the coordinates across the profile's major-axis.

        This function computes the distance from the profile centre to the edge of this 2D grid.

        Because the centres of the galaxy's light and mass profiles can be offset from one another, thsi means the
        radially grid computed for each profile can have different shapes. Therefore plots using a `Galaxy` object
        use the biggest radial grid.

        If a 1D grid is input it returns the shape of this grid, as the grid itself defines the radial coordinates.

        Parameters
        ----------
        grid
            A 1D or 2D grid from which a 1D plot of the profile is to be created.
        """
        return max(
            [
                profile.radial_projected_shape_slim_from(grid=grid)
                for key, profile in self.profile_dict.items()
            ]
        )

    def grid_radial_from(self, grid, centre, angle):

        if isinstance(grid, aa.Grid1D) or isinstance(grid, aa.Grid2DIrregular):
            return grid

        radial_projected_shape_slim = self.radial_projected_shape_slim_from(grid=grid)

        return grid.grid_2d_radial_projected_from(
            centre=centre, angle=angle + 90, shape_slim=radial_projected_shape_slim
        )

    @aa.grid_dec.grid_2d_to_structure
    def image_2d_from(self, grid: aa.type.Grid2DLike) -> Union[np.ndarray, aa.Array2D]:
        """
        Returns the summed 2D image of the galaxy's light profiles from a 2D grid of Cartesian (y,x) coordinates.

        If the galaxy has no light profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.light_profiles` package for details of how images are computed from a light 
        profile. 

        The decorator `grid_2d_to_structure` converts the output arrays from ndarrays to an `Array2D` data structure
        using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the image are evaluated.
        """
        if self.has_light_profile:
            return sum(
                map(lambda p: p.image_2d_from(grid=grid), self.light_profile_list)
            )
        return np.zeros((grid.shape[0],))

    def image_2d_list_from(self, grid: aa.type.Grid2DLike) -> List[aa.Array2D]:
        """
        Returns a list of the 2D images of the galaxy's light profiles from a 2D grid of Cartesian (y,x) coordinates.

        This function is primarily used in the `autogalaxy.operate.image` package, to output images of the `Galaxy`
        that have operations such as a 2D convolution or Fourier transform applied to them.

        If the galaxy has no light profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.light_profiles` package for details of how images are computed from a light
        profile.


        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the image are evaluated.
        """
        return list(map(lambda p: p.image_2d_from(grid=grid), self.light_profile_list))

    @aa.grid_dec.grid_1d_output_structure
    def image_1d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the summed 1D image of the galaxy's light profiles using a grid of Cartesian (y,x) coordinates.

        If the galaxy has no light profiles, a grid of zeros is returned.

        See `profiles.light_profiles` module for details of how this is performed.

        The decorator `grid_1d_output_structure` converts the output arrays from ndarrays to an `Array1D` data 
        structure using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 1D (x,) coordinates where values of the image are evaluated.
        """
        if self.has_light_profile:

            image_1d_list = []

            for light_profile in self.light_profile_list:

                grid_radial = self.grid_radial_from(
                    grid=grid, centre=light_profile.centre, angle=light_profile.angle
                )

                image_1d_list.append(light_profile.image_1d_from(grid=grid_radial))

            return sum(image_1d_list)

        return np.zeros((grid.shape[0],))

    @property
    def mass_profile_list(self) -> List[MassProfile]:
        return [value for value in self.__dict__.values() if is_mass_profile(value)]

    @property
    def has_mass_profile(self) -> bool:
        return len(self.mass_profile_list) > 0

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the summed 2D deflection angles of the galaxy's mass profiles from a 2D grid of Cartesian (y,x) 
        coordinates.

        If the galaxy has no mass profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.mass_profiles` package for details of how deflection angles are computed from a 
        mass profile. 

        The decorator `grid_2d_to_vector_yx` converts the output arrays from ndarrays to a `VectorYX2D` data structure
        using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the deflection angles are evaluated.
        """
        if self.has_mass_profile:
            return sum(
                map(
                    lambda p: p.deflections_yx_2d_from(grid=grid),
                    self.mass_profile_list,
                )
            )
        return np.zeros((grid.shape[0], 2))

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the summed 2D convergence of the galaxy's mass profiles from a 2D grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.mass_profiles` package for details of how convergences are computed from a mass 
        profile. 

        The decorator `grid_2d_to_structure` converts the output arrays from ndarrays to an `Array2D` data structure
        using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the convergence are evaluated.
        """
        if self.has_mass_profile:
            return sum(
                map(lambda p: p.convergence_2d_from(grid=grid), self.mass_profile_list)
            )
        return np.zeros((grid.shape[0],))

    @aa.grid_dec.grid_1d_output_structure
    def convergence_1d_from(self, grid: aa.type.Grid1D2DLike) -> np.ndarray:
        """
        Returns the summed 1D convergence of the galaxy's mass profiles using a grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a grid of zeros is returned.

        See `profiles.mass_profiles` module for details of how this is performed.

        The decorator `grid_1d_output_structure` converts the output arrays from ndarrays to an `Array1D` data 
        structure using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 1D (x,) coordinates where values of the convergence are evaluated.
        """
        if self.has_mass_profile:

            convergence_1d_list = []

            for mass_profile in self.mass_profile_list:

                grid_radial = self.grid_radial_from(
                    grid=grid, centre=mass_profile.centre, angle=mass_profile.angle
                )

                convergence_1d_list.append(
                    mass_profile.convergence_1d_from(grid=grid_radial)
                )

            return sum(convergence_1d_list)

        return np.zeros((grid.shape[0],))

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the summed 2D potential of the galaxy's mass profiles from a 2D grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a numpy array of zeros is returned.

        See the `autogalaxy.profiles.mass_profiles` package for details of how potentials are computed from a mass 
        profile. 

        The decorator `grid_2d_to_structure` converts the output arrays from ndarrays to an `Array2D` data structure
        using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the potential are evaluated.
        """
        if self.has_mass_profile:
            return sum(
                map(lambda p: p.potential_2d_from(grid=grid), self.mass_profile_list)
            )
        return np.zeros((grid.shape[0],))

    @aa.grid_dec.grid_1d_output_structure
    def potential_1d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the summed 1D potential of the galaxy's mass profiles using a grid of Cartesian (y,x) coordinates.

        If the galaxy has no mass profiles, a grid of zeros is returned.

        See `profiles.mass_profiles` module for details of how this is performed.

        The decorator `grid_1d_output_structure` converts the output arrays from ndarrays to an `Array1D` data 
        structure using the input `grid`'s attributes.

        Parameters
        ----------
        grid
            The 1D (x,) coordinates where values of the potential are evaluated.
        """
        if self.has_mass_profile:

            potential_1d_list = []

            for mass_profile in self.mass_profile_list:

                grid_radial = self.grid_radial_from(
                    grid=grid, centre=mass_profile.centre, angle=mass_profile.angle
                )

                potential_1d_list.append(
                    mass_profile.potential_1d_from(grid=grid_radial)
                )

            return sum(potential_1d_list)

        return np.zeros((grid.shape[0],))

    @property
    def has_pixelization(self) -> bool:
        return self.pixelization is not None

    @property
    def has_regularization(self) -> bool:
        return self.regularization is not None

    @property
    def has_hyper_galaxy(self) -> bool:
        return self.hyper_galaxy is not None

    @property
    def contribution_map(self) -> aa.Array2D:
        """
        Returns the contribution map of a galaxy, which represents the fraction of
        flux in each pixel that the galaxy is attributed to contain, hyper to the
        *contribution_factor* hyper_galaxies-parameter.

        This is computed by dividing that galaxy's flux by the total flux in that \
        pixel and then scaling by the maximum flux such that the contribution map \
        ranges between 0 and 1.

        Parameters
        -----------

        """
        return self.hyper_galaxy.contribution_map_from(
            hyper_model_image=self.hyper_model_image,
            hyper_galaxy_image=self.hyper_galaxy_image,
        )

    @property
    def half_light_radius(self):
        return None

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class and its children profiles in the the galaxy as a `ValueIrregular`
        or `Grid2DIrregular` object.

        For example, if a galaxy has two light profiles and we want the `LightProfile` axis-ratios, the following:

        `galaxy.extract_attribute(cls=LightProfile, name="axis_ratio"`

        would return:

        ValuesIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want the `MassProfile` centres, the following:

        `galaxy.extract_attribute(cls=MassProfile, name="centres"`

         would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all light profiles colored by their profile.
        """

        def extract(value, name):

            try:
                return getattr(value, name)
            except (AttributeError, IndexError):
                return None

        attributes = [
            extract(value, attr_name)
            for value in self.__dict__.values()
            if isinstance(value, cls)
        ]

        attributes = list(filter(None, attributes))

        if attributes == []:
            return None
        elif isinstance(attributes[0], float):
            return aa.ValuesIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return aa.Grid2DIrregular(grid=attributes)

    def luminosity_within_circle_from(self, radius: float):
        """
        Returns the total luminosity of the galaxy's light profiles within a circle of specified radius.

            See *light_profile_list.luminosity_within_circle* for details of how this is performed.

            Parameters
            ----------
            radius
                The radius of the circle to compute the dimensionless mass within.
            unit_luminosity
                The unit_label the luminosity is returned in {esp, counts}.
            exposure_time
                The exposure time of the observation, which converts luminosity from electrons per second unit_label to counts.
        """
        if self.has_light_profile:
            return sum(
                map(
                    lambda p: p.luminosity_within_circle_from(radius=radius),
                    self.light_profile_list,
                )
            )

    def mass_angular_within_circle_from(self, radius: float):
        """
        Integrate the mass profiles's convergence profile to compute the total mass within a circle of \
        specified radius. This is centred on the mass profile.

        The following unit_label for mass can be specified and output:

        - Dimensionless angular unit_label (default) - 'angular'.
        - Solar masses - 'angular' (multiplies the angular mass by the critical surface mass density).

        Parameters
        ----------
        radius : dim.Length
            The radius of the circle to compute the dimensionless mass within.
        unit_mass
            The unit_label the mass is returned in {angular, angular}.
        critical_surface_density or None
            The critical surface mass density of the strong lens configuration, which converts mass from angulalr \
            unit_label to phsical unit_label (e.g. solar masses).
        """
        if self.has_mass_profile:
            return sum(
                map(
                    lambda p: p.mass_angular_within_circle_from(radius=radius),
                    self.mass_profile_list,
                )
            )
        else:
            raise exc.GalaxyException(
                "You cannot perform a mass-based calculation on a galaxy which does not have a mass-profile"
            )

    @property
    def profile_dict(self) -> Dict:
        return {
            key: value
            for key, value in self.__dict__.items()
            if isinstance(value, GeometryProfile)
        }


class HyperGalaxy:
    _ids = count()

    def __init__(
        self,
        contribution_factor: float = 0.0,
        noise_factor: float = 0.0,
        noise_power: float = 1.0,
    ):
        """
        If a `Galaxy` is given a *HyperGalaxy* as an attribute, the noise-map in \
        the regions of the image that the galaxy is located will be hyper, to prevent \
        over-fitting of the galaxy.
        
        This is performed by first computing the hyper_galaxies-galaxy's 'contribution-map', \
        which determines the fraction of flux in every pixel of the image that can be \
        associated with this particular hyper_galaxies-galaxy. This is computed using \
        hyper_galaxies-hyper_galaxies set (e.g. fitting.fit_data.FitDataHyper), which includes  best-fit \
        unblurred_image_1d of the galaxy's light from a previous analysis search.
         
        The *HyperGalaxy* class contains the hyper_galaxies-parameters which are associated \
        with this galaxy for scaling the noise-map.
        
        Parameters
        -----------
        contribution_factor
            Factor that adjusts how much of the galaxy's light is attributed to the
            contribution map.
        noise_factor
            Factor by which the noise-map is increased in the regions of the galaxy's
            contribution map.
        noise_power
            The power to which the contribution map is raised when scaling the
            noise-map.
        """
        self.contribution_factor = contribution_factor
        self.noise_factor = noise_factor
        self.noise_power = noise_power

        self.component_number = next(self._ids)

    def __eq__(self, other):
        if isinstance(other, HyperGalaxy):
            return (
                self.contribution_factor == other.contribution_factor
                and self.noise_factor == other.noise_factor
                and self.noise_power == other.noise_power
            )
        return False

    def __str__(self):
        return "\n".join(["{}: {}".format(k, v) for k, v in self.__dict__.items()])

    def contribution_map_from(self, hyper_model_image, hyper_galaxy_image):
        """
        Returns the contribution map of a galaxy, which represents the fraction of
        flux in each pixel that the galaxy is attributed to contain, hyper to the
        *contribution_factor* hyper_galaxies-parameter.

        This is computed by dividing that galaxy's flux by the total flux in that \
        pixel and then scaling by the maximum flux such that the contribution map \
        ranges between 0 and 1.

        Parameters
        -----------
        hyper_model_image
            The best-fit model image to the observed image from a previous analysis
            search. This provides the total light attributed to each image pixel by the
            model.
        hyper_galaxy_image
            A model image of the galaxy (from light profiles or an inversion) from a
            previous analysis search.
        """
        contribution_map = np.divide(
            hyper_galaxy_image, np.add(hyper_model_image, self.contribution_factor)
        )
        return np.divide(contribution_map, np.max(contribution_map))

    def hyper_noise_map_via_hyper_images_from(
        self,
        hyper_model_image: aa.Array2D,
        hyper_galaxy_image: aa.Array2D,
        noise_map: aa.Array2D,
    ) -> aa.Array2D:
        contribution_map = self.contribution_map_from(
            hyper_model_image=hyper_model_image, hyper_galaxy_image=hyper_galaxy_image
        )

        return self.hyper_noise_map_from(
            noise_map=noise_map, contribution_map=contribution_map
        )

    def hyper_noise_map_from(
        self, noise_map: aa.Array2D, contribution_map: aa.Array2D
    ) -> aa.Array2D:
        """
        Returns a hyper galaxy hyper_galaxies noise-map from a baseline noise-map.

            This uses the galaxy contribution map and the *noise_factor* and *noise_power*
            hyper_galaxies-parameters.

            Parameters
            -----------
            noise_map
                The observed noise-map (before scaling).
            contribution_map
                The galaxy contribution map.
        """
        return self.noise_factor * (noise_map * contribution_map) ** self.noise_power


class Redshift(float):
    def __new__(cls, redshift):
        # noinspection PyArgumentList
        return float.__new__(cls, redshift)

    def __init__(self, redshift):
        float.__init__(redshift)
