import numpy as np
from typing import Dict, List, Optional, Tuple, Type, Union

import autoarray as aa

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.profiles.light.basis import Basis
from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.light.linear import LightProfileLinear
from autogalaxy.profiles.light.snr import LightProfileSNR
from autogalaxy.operate.image import OperateImageGalaxies
from autogalaxy.operate.deflections import OperateDeflections

from autogalaxy import exc


class Galaxies(List, OperateImageGalaxies, OperateDeflections):
    def __init__(
        self,
        galaxies: List[Galaxy],
        run_time_dict: Optional[Dict] = None,
    ):
        """
        A collection of galaxies, used to perform operations on the galaxies as a group.

        It is common for a user to have multiple galaxies in a list, for which they may perform operations like
        creating the image of the light profiles or all galaxies or the potential of the mass profiles of all galaxies.
        
        Many of these calculations are straight forward, for example for the image of all galaxies we simply sum the
        images of each galaxy. 
        
        However, there are more complex operations that can be performed on the galaxies as a group, for example
        computing the blured image of a group of galaxies where some galaxies have operated light profiles (meaning 
        that PSF blurring is already applied to them and thus should be skipped) and another subset of galaxies have 
        normal light profiles (meaning that PSF blurring should be applied to them).
        
        This calculation requires a careful set of operations to ensure that the PSF blurring is only applied to the
        subset of galaxies that do not have operated light profiles. This is an example of a calculation that is
        performed by the `Galaxies` class, simplifing the code required to perform this operation.
        
        Users may find that they often omit the `Galaxies` class and instead perform these operations in a more manual
        way. This is fine, but care must be taken to ensure that the operations are performed correctly.
        
        Parameters
        ----------
        galaxies
            The list of galaxies whose calculations are performed by this class.
        run_time_dict
            A dictionary of information on the run-times of function calls, including the total time and time spent on
            different calculations.
        """

        super().__init__(galaxies)
        self.run_time_dict = run_time_dict

    def image_2d_list_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> List[aa.Array2D]:
        """
        Returns a list of the 2D images for each galaxy from a 2D grid of Cartesian (y,x) coordinates.

        The image of each galaxy is computed by summing the images of all light profiles in that galaxy. If a galaxy 
        has no light profiles, a numpy array of zeros is returned.

        For example, if there are 3 galaxies and only the first two have light profiles, the returned list of images 
        will be the image of the first two galaxies. The image of the third galaxies will be a numpy array of zeros.

        The images output by this function do not include instrument operations, such as PSF convolution (for imaging
        data) or a Fourier transform (for interferometer data).

        Inherited methods in the `autogalaxy.operate.image` package can apply these operations to the images. 
        These functions may have the `operated_only` input passed to them, which is why this function includes 
        the `operated_only` input.

        If the `operated_only` input is included, the function omits light profiles which are parents of
        the `LightProfileOperated` object, which signifies that the light profile represents emission that has
        already had the instrument operations (e.g. PSF convolution, a Fourier transform) applied to it and therefore
        that operation is not performed again.

        See the `autogalaxy.profiles.light` package for details of how images are computed from a light
        profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the image are evaluated.
        operated_only
            The returned list from this function contains all light profile images, and they are never operated on
            (e.g. via the imaging PSF). However, inherited methods in the `autogalaxy.operate.image` package can
            apply these operations to the images, which may have the `operated_only` input passed to them. This input
            therefore is used to pass the `operated_only` input to these methods.
        """
        return [
            galaxy.image_2d_from(grid=grid, operated_only=operated_only)
            for galaxy in self
        ]

    @aa.grid_dec.grid_2d_to_structure
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> aa.Array2D:
        """
        Returns the 2D image of all galaxies summed from a 2D grid of Cartesian (y,x) coordinates.

        This function first computes the image of each galaxy, via the function `image_2d_list_from`. The
        images are then summed to give the overall image of the galaxies.

        Refer to the function `image_2d_list_from` for a full description of the calculation and how the `operated_only`
        input is used.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the image are evaluated.
        operated_only
            The returned list from this function contains all light profile images, and they are never operated on
            (e.g. via the imaging PSF). However, inherited methods in the `autogalaxy.operate.image` package can
            apply these operations to the images, which may have the `operated_only` input passed to them. This input
            therefore is used to pass the `operated_only` input to these methods.
        """
        return sum(self.image_2d_list_from(grid=grid, operated_only=operated_only))

    def galaxy_image_2d_dict_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> {Galaxy: np.ndarray}:
        """
        Returns a dictionary associating every `Galaxy` object with its corresponding 2D image, using the instance 
        of each galaxy as the dictionary keys.

        This object is used for adaptive-features, which use the image of each galaxy in a model-fit in order to
        adapt quantities like a pixelization or regularization scheme to the surface brightness of the galaxies being
        fitted.

        By inheriting from `OperateImageGalaxies` functions which apply operations of this dictionary are accessible,
        for example convolving every image with a PSF or applying a Fourier transform to create a galaxy-visibilities
        dictionary.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.

        Returns
        -------
        A dictionary associated every galaxy with its corresponding 2D image.
        """

        galaxy_image_2d_dict = dict()

        image_2d_list = self.image_2d_list_from(grid=grid, operated_only=operated_only)

        for galaxy_index, galaxy in enumerate(self):
            galaxy_image_2d_dict[galaxy] = image_2d_list[galaxy_index]

        return galaxy_image_2d_dict

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the summed 2D deflections angles of all galaxies from a 2D grid of Cartesian (y,x) coordinates.

        The deflections of each galaxy is computed by summing the deflections of all mass profiles in that galaxy. If a
        galaxy has no mass profiles, a numpy array of zeros is returned.

        This calculation does not account for multi-plane ray-tracing effects, it is simply the sum of the deflections
        of all galaxies. The `Tracer` class in PyAutoLens is required for this.

        For example, if there are 3 galaxies and only the first two have mass profiles, the returned list of deflections
        will be the deflections of the first two galaxies. The deflections of the third galaxies will be a numpy 
        array of zeros.

        See the `autogalaxy.profiles.mass` package for details of how deflections are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the deflections are evaluated.
        """
        if self:
            return sum(map(lambda g: g.deflections_yx_2d_from(grid=grid), self))
        return np.zeros(shape=(grid.shape[0], 2))

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the summed 2D convergence of all galaxies from a 2D grid of Cartesian (y,x) coordinates.

        The convergence of each galaxy is computed by summing the convergence of all mass profiles in that galaxy. If a
        galaxy has no mass profiles, a numpy array of zeros is returned.

        This calculation does not account for multi-plane ray-tracing effects, it is simply the sum of the convergence
        of all galaxies. The `Tracer` class in PyAutoLens is required for this.

        For example, if there are 3 galaxies and only the first two have mass profiles, the returned list of convergence
        will be the convergence of the first two galaxies. The convergence of the third galaxies will be a numpy 
        array of zeros.

        See the `autogalaxy.profiles.mass` package for details of how convergence are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the convergence are evaluated.
        """
        if self:
            return sum(map(lambda g: g.convergence_2d_from(grid=grid), self))
        return np.zeros((grid.shape[0],))

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the summed 2D potential of all galaxies from a 2D grid of Cartesian (y,x) coordinates.

        The potential of each galaxy is computed by summing the potential of all mass profiles in that galaxy. If a
        galaxy has no mass profiles, a numpy array of zeros is returned.

        This calculation does not account for multi-plane ray-tracing effects, it is simply the sum of the potential
        of all galaxies. The `Tracer` class in PyAutoLens is required for this.

        For example, if there are 3 galaxies and only the first two have mass profiles, the returned list of potential
        will be the potential of the first two galaxies. The potential of the third galaxies will be a numpy 
        array of zeros.

        See the `autogalaxy.profiles.mass` package for details of how potential are computed from a mass profile.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the potential are evaluated.
        """
        if self:
            return sum(map(lambda g: g.potential_2d_from(grid=grid), self))
        return np.zeros((grid.shape[0],))

    def has(self, cls: Union[Type, Tuple[Type]]) -> bool:
        """
        Returns a bool specifying whether any of the galaxies has a certain class type.

        For example, for the input `cls=ag.LightProfile`, this function returns True if any galaxy has a
        light profile and false if no galaxy has a light profile.

        This function is used to check for mass profiles and specific types of profiles, like the linear light profile.

        Parameters
        ----------
        cls
            The class type of the galaxy which is checked for in the tracer.

        Returns
        -------
        True if any galaxy in the tracer has the input class type, else False.
        """
        return any(list(map(lambda galaxy: galaxy.has(cls=cls), self)))

    def cls_list_from(self, cls: Type) -> List:
        """
        Returns a list of objects in the galaxies which are an instance of the input `cls`.

        For example:

        - If the input is `cls=ag.LightProfile`, a list containing all light profiles of all galaxies is returned.

        Returns
        -------
            The list of objects in the plane that inherit from input `cls`.
        """
        cls_list = []

        for galaxy in self:
            if galaxy.has(cls=cls):
                for cls_galaxy in galaxy.cls_list_from(cls=cls):
                    cls_list.append(cls_galaxy)

        return cls_list

    def extract_attribute(self, cls, attr_name):
        """
        Returns an attribute of a class in all galaxies as a `ValueIrregular` or `Grid2DIrregular` object.

        For example, if there is one galaxy with two light profiles and we want its axis-ratios, the following:

        `plane.extract_attribute(cls=LightProfile, name="axis_ratio")`

        would return:

        ArrayIrregular(values=[axis_ratio_0, axis_ratio_1])

        If a galaxy has three mass profiles and we want their centres, the following:

        `plane.extract_attribute(cls=MassProfile, name="centres")`

        would return:

        GridIrregular2D(grid=[(centre_y_0, centre_x_0), (centre_y_1, centre_x_1), (centre_y_2, centre_x_2)])

        This is used for visualization, for example plotting the centres of all mass profiles colored by their profile.
        """

        def extract(value, name):
            try:
                return getattr(value, name)
            except (AttributeError, IndexError):
                return None

        attributes = [
            extract(value, attr_name)
            for galaxy in self
            for value in galaxy.__dict__.values()
            if isinstance(value, cls)
        ]

        if attributes == []:
            return None
        elif isinstance(attributes[0], float):
            return aa.ArrayIrregular(values=attributes)
        elif isinstance(attributes[0], tuple):
            return aa.Grid2DIrregular(values=attributes)

    @property
    def perform_inversion(self) -> bool:
        """
        Returns a bool specifying whether this fit object performs an inversion.

        This is based on whether any of the galaxies have a `Pixelization` or `LightProfileLinear` object, in which
        case an inversion is performed.

        Returns
        -------
            A bool which is True if an inversion is performed.
        """
        if self.has(cls=(aa.Pixelization, LightProfileLinear)):
            return True
        elif self.has(cls=Basis):
            basis_list = self.cls_list_from(cls=Basis)
            for basis in basis_list:
                for light_profile in basis.light_profile_list:
                    if isinstance(light_profile, LightProfileLinear):
                        return True

        return False
