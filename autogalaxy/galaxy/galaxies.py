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
        galaxies : List[Galaxy],
        run_time_dict: Optional[Dict] = None,
    ):
        """
        A collection of galaxies, which can be used to perform operations on the galaxies as a group.

        Parameters
        ----------
        redshift or None
            The redshift of the plane.
        galaxies : [Galaxy]
            The list of galaxies in this plane.
        """

        super().__init__(galaxies)
        self.run_time_dict = run_time_dict

    @aa.grid_dec.grid_2d_to_structure
    def image_2d_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> aa.Array2D:
        """
        Returns the profile-image plane image of the list of galaxies of the plane's sub-grid, by summing the
        individual images of each galaxy's light profile.

        If the `operated_only` input is included, the function omits light profiles which are parents of
        the `LightProfileOperated` object, which signifies that the light profile represents emission that has
        already had the instrument operations (e.g. PSF convolution, a Fourier transform) applied to it.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Parameters
        ----------
        grid
            The 2D (y, x) coordinates where values of the image are evaluated.
        operated_only
            By default, the image is the sum of light profile images (irrespective of whether they have been operatd on
            or not). If this input is included as a bool, only images which are or are not already operated are summed
            and returned.
        """
        return sum(self.image_2d_list_from(grid=grid, operated_only=operated_only))

    def image_2d_list_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> List[aa.Array2D]:
        return [
            galaxy.image_2d_from(grid=grid, operated_only=operated_only)
            for galaxy in self
        ]

    def galaxy_image_2d_dict_from(
        self, grid: aa.type.Grid2DLike, operated_only: Optional[bool] = None
    ) -> {Galaxy: np.ndarray}:
        """
        Returns a dictionary associating every `Galaxy` object in the `Plane` with its corresponding 2D image, using
        the instance of each galaxy as the dictionary keys.

        This object is used for adapt features, which use the image of each galaxy in a model-fit in order to
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
        A dictionary associated every galaxy in the plane with its corresponding 2D image.
        """

        galaxy_image_2d_dict = dict()

        image_2d_list = self.image_2d_list_from(grid=grid, operated_only=operated_only)

        for galaxy_index, galaxy in enumerate(self):
            galaxy_image_2d_dict[galaxy] = image_2d_list[galaxy_index]

        return galaxy_image_2d_dict


    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        if self:
            return sum(
                map(lambda g: g.deflections_yx_2d_from(grid=grid), self)
            )
        return np.zeros(shape=(grid.shape[0], 2))

    @aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the convergence of the list of galaxies of the plane's sub-grid, by summing the individual convergences \
        of each galaxy's mass profile.

        The convergence is calculated on the sub-grid and binned-up to the original grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is `True`.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Internally data structures are treated as ndarrays, however the decorator `grid_2d_to_structure` converts
        the output to an `Array2D` using the input `grid`'s attributes.

        Parameters
        ----------
        grid : Grid2D
            The grid (or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
            potential is calculated on.
        galaxies : [Galaxy]
            The galaxies whose mass profiles are used to compute the surface densities.
        """
        if self:
            return sum(map(lambda g: g.convergence_2d_from(grid=grid), self))
        return np.zeros((grid.shape[0],))

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        """
        Returns the potential of the list of galaxies of the plane's sub-grid, by summing the individual potentials \
        of each galaxy's mass profile.

        The potential is calculated on the sub-grid and binned-up to the original grid by taking the mean
        value of every set of sub-pixels, provided the *returned_binned_sub_grid* bool is `True`.

        If the plane has no galaxies (or no galaxies have mass profiles) an arrays of all zeros the shape of the plane's
        sub-grid is returned.

        Internally data structures are treated as ndarrays, however the decorator `grid_2d_to_structure` converts
        the output to an `Array2D` using the input `grid`'s attributes.

        Parameters
        ----------
        grid : Grid2D
            The grid (or sub) of (y,x) arc-second coordinates at the centre of every unmasked pixel which the \
            potential is calculated on.
        galaxies : [Galaxy]
            The galaxies whose mass profiles are used to compute the surface densities.
        """
        if self:
            return sum(map(lambda g: g.potential_2d_from(grid=grid), self))
        return np.zeros((grid.shape[0],))

    def has(self, cls: Union[Type, Tuple[Type]]) -> bool:
        return any(list(map(lambda galaxy: galaxy.has(cls=cls), self)))

    def cls_list_from(self, cls: Type) -> List:
        """
        Returns a list of objects in the plane which are an instance of the input `cls`.

        For example:

        - If the input is `cls=ag.LightProfile`, a list containing all light profiles in the plane is returned.

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
        Returns an attribute of a class in `Plane` as a `ValueIrregular` or `Grid2DIrregular` object.

        For example, if a plane has a galaxy which two light profiles and we want its axis-ratios, the following:

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

        This is based on whether any of the galaxies in the `model_obj` have a `Pixelization` or `LightProfileLinear`
        object, in which case an inversion is performed.

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