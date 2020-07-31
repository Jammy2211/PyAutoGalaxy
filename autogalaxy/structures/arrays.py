import logging

from autoarray.structures.arrays import arrays
from autoarray.util import array_util

logging.basicConfig()
logger = logging.getLogger(__name__)


class Array(arrays.Array):
    @classmethod
    def from_fits(
        cls,
        file_path,
        hdu=0,
        pixel_scales=None,
        sub_size=1,
        origin=(0.0, 0.0),
        store_in_1d=True,
    ):
        """Create an Array (see *AbstractArray.__new__*) by loaing the array values from a .fits file.

        Parameters
        ----------
        file_path : str
            The path the file is output to, including the filename and the '.fits' extension,
            e.g. '/path/to/filename.fits'
        hdu : int
            The Header-Data Unit of the .fits file the array data is loaded from.
        pixel_scales : (float, float) or float
            The pixel conversion scale of a pixel in the y and x directions. If input as a float, the pixel_scales
            are converted to the format (float, float).
        sub_size : int
            The size (sub_size x sub_size) of each unmasked pixels sub-array.
        origin : (float, float)
            The origin of the array's mask.
        store_in_1d : bool
            If True, the array is stored in 1D as an ndarray of shape [total_unmasked_pixels]. If False, it is
            stored in 2D as an ndarray of shape [total_y_pixels, total_x_pixels].
        """
        array_2d = array_util.numpy_array_2d_from_fits(
            file_path=file_path, hdu=hdu, flip_for_ds9=True
        )
        return cls.manual_2d(
            array=array_2d,
            pixel_scales=pixel_scales,
            sub_size=sub_size,
            origin=origin,
            store_in_1d=store_in_1d,
        )
