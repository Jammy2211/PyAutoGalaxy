from autoarray.structures import abstract_structure
from autoarray.structures.frame import frame
from autoarray.util import array_util


class Frame(frame.Frame):
    @classmethod
    def from_fits(
        cls,
        file_path,
        hdu,
        roe_corner=(1, 0),
        scans=None,
        exposure_info=None,
        pixel_scales=None,
    ):
        """Load the image ci_data from a fits file.

        Params
        ----------
        path : str
            The path to the ci_data
        filename : str
            The file phase_name of the fits image ci_data.
        hdu : int
            The HDU number in the fits file containing the image ci_data.
        frame_geometry : FrameArray.FrameGeometry
            The geometry of the ci_frame, defining the direction of parallel and serial clocking and the \
            locations of different scans of the CCD (overscans, prescan, etc.)
        """

        pixel_scales = abstract_structure.convert_pixel_scales(
            pixel_scales=pixel_scales
        )

        array = array_util.numpy_array_2d_from_fits(
            file_path=file_path, hdu=hdu, flip_for_ds9=True
        )

        return cls.manual(
            array=array,
            roe_corner=roe_corner,
            scans=scans,
            exposure_info=exposure_info,
            pixel_scales=pixel_scales,
        )
