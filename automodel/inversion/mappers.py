import autoarray as aa
from automodel.util import mapper_util

import numpy as np


class Mapper(object):
    def __init__(self, pixels, grid, pixel_centres, geometry, hyper_image=None):
        """ Abstract base class representing a mapper, which maps unmasked pixels on a masked 2D array (in the form of \
        a grid, see the *hyper_galaxies.array.grid* module) to discretized pixels in a pixelization.

        1D structures are used to represent these mappings, for example between the different grid in a grid \
        (e.g. the / sub grid). This follows the syntax grid_to_grid, whereby the index of a value on one grid \
        equals that of another grid, for example:

        - image_to_pix[2] = 1  tells us that the 3rd pixel on a grid maps to the 2nd pixel of a pixelization.
        - sub_to_pix4] = 2  tells us that the 5th sub-pixel of a sub-grid maps to the 3rd pixel of a pixelization.
        - pix_to_image[2] = 5 tells us that the 3rd pixel of a pixelization maps to the 6th (unmasked) pixel of a \
                            grid.

        Parameters
        ----------
        pixels : int
            The number of pixels in the mapper's pixelization.
        grid: gridStack
            A stack of grid's which are mapped to the pixelization (includes an and sub grid).
        border : grid.GridBorder
            The border of the grid's grid.
        hyper_image : ndarray
            A pre-computed hyper_galaxies-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """
        self.pixels = pixels
        self.grid = grid
        self.pixel_centres = pixel_centres
        self.geometry = geometry
        self.hyper_image = hyper_image

    @property
    def mapping_matrix(self):
        """The util matrix is a matrix representing the util between every unmasked pixel of a grid and \
        the pixels of a pixelization. Non-zero entries signify a util, whereas zeros signify no util.

        For example, if the grid has 5 pixels and the pixelization 3 pixels, with the following mappings:

        pixel 0 -> pixelization pixel 0
        pixel 1 -> pixelization pixel 0
        pixel 2 -> pixelization pixel 1
        pixel 3 -> pixelization pixel 1
        pixel 4 -> pixelization pixel 2

        The util matrix (which is of dimensions regular_pixels x pixelization_pixels) would appear as follows:

        [1, 0, 0] [0->0]
        [1, 0, 0] [1->0]
        [0, 1, 0] [2->1]
        [0, 1, 0] [3->1]
        [0, 0, 1] [4->2]

        The util matrix is in fact built using the sub-grid of the grid, whereby each pixel is \
        divided into a grid of sub-pixels which are all paired to pixels in the pixelization. The entires \
        in the util matrix now become fractional values dependent on the sub-grid size. For example, for a 2x2 \
        sub-grid in each pixel (which means the fraction value is 1.0/(2.0^2) = 0.25, if we have the following mappings:

        pixel 0 -> sub pixel 0 -> pixelization pixel 0
        pixel 0 -> sub pixel 1 -> pixelization pixel 1
        pixel 0 -> sub pixel 2 -> pixelization pixel 1
        pixel 0 -> sub pixel 3 -> pixelization pixel 1
        pixel 1 -> sub pixel 0 -> pixelization pixel 1
        pixel 1 -> sub pixel 1 -> pixelization pixel 1
        pixel 1 -> sub pixel 2 -> pixelization pixel 1
        pixel 1 -> sub pixel 3 -> pixelization pixel 1
        pixel 2 -> sub pixel 0 -> pixelization pixel 2
        pixel 2 -> sub pixel 1 -> pixelization pixel 2
        pixel 2 -> sub pixel 2 -> pixelization pixel 3
        pixel 2 -> sub pixel 3 -> pixelization pixel 3

        The util matrix (which is still of dimensions regular_pixels x source_pixels) would appear as follows:

        [0.25, 0.75, 0.0, 0.0] [1 sub-pixel maps to pixel 0, 3 map to pixel 1]
        [ 0.0,  1.0, 0.0, 0.0] [All sub-pixels map to pixel 1]
        [ 0.0,  0.0, 0.5, 0.5] [2 sub-pixels map to pixel 2, 2 map to pixel 3]
        """
        return mapper_util.mapping_matrix_from_pixelization_1d_index_for_sub_mask_1d_index(
            pixelization_1d_index_for_sub_mask_1d_index=self.pixelization_1d_index_for_sub_mask_1d_index,
            pixels=self.pixels,
            total_mask_pixels=self.grid.mask.pixels_in_mask,
            mask_1d_index_for_sub_mask_1d_index=self._mask_1d_index_for_sub_mask_1d_index,
            sub_fraction=self.grid.mask.sub_fraction,
        )

    @property
    def pixel_neighbors(self):
        return self.geometry.pixel_neighbors

    @property
    def pixel_neighbors_size(self):
        return self.geometry.pixel_neighbors_size

    @property
    def _mask_1d_index_for_sub_mask_1d_index(self):
        return self.grid.mask.regions._mask_1d_index_for_sub_mask_1d_index

    @property
    def pixelization_1d_index_for_sub_mask_1d_index(self):
        raise NotImplementedError("pixelization_1d_index_for_sub_mask_1d_index should be overridden")

    @property
    def all_sub_mask_1d_indexes_for_pixelization_1d_index(self):
        """Compute the mappings between a pixelization's pixels and the unmasked sub-grid pixels. These mappings \
        are determined after the grid is used to determine the pixelization.

        The pixelization's pixels map to different number of sub-grid pixels, thus a list of lists is used to \
        represent these mappings"""

        all_sub_mask_1d_indexes_for_pixelization_1d_index = [
            [] for _ in range(self.pixels)
        ]

        for mask_1d_index, pixelization_1d_index in enumerate(
            self.pixelization_1d_index_for_sub_mask_1d_index
        ):
            all_sub_mask_1d_indexes_for_pixelization_1d_index[
                pixelization_1d_index
            ].append(mask_1d_index)

        return all_sub_mask_1d_indexes_for_pixelization_1d_index

    def pixel_signals_from_signal_scale(
        self, signal_scale
    ):

        return mapper_util.adaptive_pixel_signals_from_images(
            pixels=self.pixels,
            signal_scale=signal_scale,
            pixelization_1d_index_for_sub_mask_1d_index=self.pixelization_1d_index_for_sub_mask_1d_index,
            mask_1d_index_for_sub_mask_1d_index=self.grid.regions._mask_1d_index_for_sub_mask_1d_index,
            hyper_image=self.hyper_image,
        )

class RectangularMapper(Mapper):
    def __init__(
        self, pixels, grid, pixel_centres, shape_2d, geometry, hyper_image=None
    ):
        """ Class representing a rectangular mapper, which maps unmasked pixels on a masked 2D array (in the form of \
        a grid, see the *hyper_galaxies.array.grid* module) to pixels discretized on a rectangular grid.

        The and uniform geometry of the rectangular grid is used to perform efficient pixel pairings.

        Parameters
        ----------
        pixels : int
            The number of pixels in the rectangular pixelization (y_pixels*x_pixels).
        grid : gridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        border : grid.GridBorder
            The border of the grid's grid.
        shape_2d : (int, int)
            The dimensions of the rectangular grid of pixels (y_pixels, x_pixel)
        geometry : pixelization.Rectangular.Geometry
            The geometry (e.g. y / x edge locations, pixel-scales) of the rectangular pixelization.
        """
        self.shape_2d = shape_2d
        super(RectangularMapper, self).__init__(
            pixels=pixels,
            grid=grid,
            pixel_centres=pixel_centres,
            geometry=geometry,
            hyper_image=hyper_image,
        )

    @property
    def is_image_plane_pixelization(self):
        return False

    @property
    def pixelization_1d_index_for_sub_mask_1d_index(self):
        """The 1D index mappings between the sub grid's pixels and rectangular pixelization's pixels"""
        return aa.util.grid.grid_pixel_indexes_1d_from_grid_arcsec_1d_shape_2d_and_pixel_scales(
            grid_arcsec_1d=self.grid, shape_2d=self.geometry.shape_2d, pixel_scales=self.geometry.pixel_scales, origin=self.geometry.origin).astype('int')

    def reconstructed_pixelization_from_solution_vector(self, solution_vector):
        """Given the solution vector of an inversion (see *inversions.Inversion*), determine the reconstructed \
        pixelization of the rectangular pixelization by using the mapper."""
        recon = aa.util.array.sub_array_2d_from_sub_array_1d(
            sub_array_1d=solution_vector,
            mask=np.full(fill_value=False, shape=self.shape_2d),
            sub_size=1,
        )
        return aa.array.manual_2d(
            array=recon, sub_size=1, pixel_scales=self.geometry.pixel_scales, origin=self.geometry.origin
        )


class VoronoiMapper(Mapper):
    def __init__(
        self, pixels, grid, pixel_centres, voronoi, geometry, hyper_image=None
    ):
        """Class representing a Voronoi mapper, which maps unmasked pixels on a masked 2D array (in the form of \
        a grid, see the *hyper_galaxies.array.grid* module) to pixels discretized on a Voronoi grid.

        The irand non-uniform geometry of the Voronoi grid means efficient pixel pairings requires knowledge \
        of how different grid map to one another.

        Parameters
        ----------
        pixels : int
            The number of pixels in the Voronoi pixelization.
        grid : gridStack
            A stack of grid describing the observed image's pixel coordinates (e.g. an image-grid, sub-grid, etc.).
        border : grid.GridBorder
            The border of the grid's grid.
        voronoi : scipy.spatial.Voronoi
            Class storing the Voronoi grid's 
        geometry : pixelization.Voronoi.Geometry
            The geometry (e.g. y / x edge locations, pixel-scales) of the Vornoi pixelization.
        hyper_image : ndarray
            A pre-computed hyper_galaxies-image of the image the mapper is expected to reconstruct, used for adaptive analysis.
        """
        self.voronoi = voronoi
        super(VoronoiMapper, self).__init__(
            pixels=pixels,
            grid=grid,
            pixel_centres=pixel_centres,
            geometry=geometry,
            hyper_image=hyper_image,
        )

    @property
    def is_image_plane_pixelization(self):
        return True

    @property
    def pixelization_1d_index_for_sub_mask_1d_index(self):
        """  The 1D index mappings between the sub pixels and Voronoi pixelization pixels. """
        return mapper_util.pixelization_1d_index_for_voronoi_sub_mask_1d_index_from_grids_and_geometry(
            grid=self.grid,
            nearest_pixelization_1d_index_for_mask_1d_index=self.pixel_centres.nearest_pixelization_1d_index_for_mask_1d_index,
            mask_1d_index_for_sub_mask_1d_index=self.grid.regions._mask_1d_index_for_sub_mask_1d_index,
            pixel_centres=self.pixel_centres,
            pixel_neighbors=self.geometry.pixel_neighbors,
            pixel_neighbors_size=self.geometry.pixel_neighbors_size,
        ).astype(
            "int"
        )
