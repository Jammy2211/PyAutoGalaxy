import numpy as np
from typing import Callable, Union

import autoarray as aa


class CalcImage:
    def __init__(self, image_2d_from: Callable):
        """
        Packages methods which manipulate the 2D image returned from the `image_2d_from` function of a light object
        (e.g. a `LightProfile`, `Galaxy`, `Plane`).

        The majority of methods apply data operators to the 2D image which perform tasks such as a 2D convolution or
        Fourier transform.

        The methods in `CalcImage` are passed to the light object to provide a concise API.

        Parameters
        ----------
        image_2d_from
            The function which returns the light object's 2D image.
        """
        self.image_2d_from = image_2d_from

    def blurred_image_2d_via_psf_from(
        self,
        grid: Union[aa.Grid2D, aa.Grid2DIterate],
        psf: aa.Kernel2D,
        blurring_grid: Union[aa.Grid2D, aa.Grid2DIterate],
    ) -> aa.Array2D:
        """
        Evaluate the light object's 2D image from a input 2D grid of coordinates and convolve it with a PSF.

        The input 2D grid may be masked, in which case values outside but near the edge of the mask will convolve light
        into the mask. A blurring grid is therefore required, which contains image pixels on the mask edge whose light
        is blurred into the light object's image by the PSF.

        The grid and blurring_grid must be a `Grid2D` objects so the evaluated image can be mapped to a uniform 2D
        array and binned up for convolution. They therefore cannot be `Grid2DIrregular` objects.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        psf
            The PSF the light object 2D image is convolved with.
        blurring_grid
            The 2D (y,x) coordinates neighboring the (masked) grid whose light is blurred into the image.
        """
        image = self.image_2d_from(grid=grid)

        blurring_image = self.image_2d_from(grid=blurring_grid)

        return psf.convolved_array_with_mask_from(
            array=image.binned.native + blurring_image.binned.native, mask=grid.mask
        )

    def blurred_image_2d_via_convolver_from(
        self,
        grid: Union[aa.Grid2D, aa.Grid2DIterate],
        convolver: aa.Convolver,
        blurring_grid: Union[aa.Grid2D, aa.Grid2DIterate],
    ) -> aa.Array2D:
        """
        Evaluate the light object's 2D image from a input 2D grid of coordinates and convolve it with a PSF, using a
        `autoarray.operators.convolver.Convolver` object. The `Convolver` object performs the 2D convolution operations
        using 1D NumPy arrays without mapping them to 2D, which is more efficient.

        The input 2D grid may be masked, in which case values outside but near the edge of the mask will convolve light
        into the mask. A blurring grid is therefore required, which contains image pixels on the mask edge whose light
        is blurred into the light object's image by the PSF.

        The grid and blurring_grid must be a `Grid2D` objects so the evaluated image can be mapped to a uniform 2D
        array and binned up for convolution. They therefore cannot be `Grid2DIrregular` objects.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        convolver
            The convolver object used perform PSF convolution on 1D numpy arrays.
        blurring_grid
            The 2D (y,x) coordinates neighboring the (masked) grid whose light is blurred into the image.
        """
        image = self.image_2d_from(grid=grid)

        blurring_image = self.image_2d_from(grid=blurring_grid)

        return convolver.convolve_image(
            image=image.binned, blurring_image=blurring_image.binned
        )

    def profile_visibilities_via_transformer_from(
        self,
        grid: Union[aa.Grid2D, aa.Grid2DIterate],
        transformer: Union[aa.TransformerDFT, aa.TransformerNUFFT],
    ) -> aa.Visibilities:
        """
        Evaluate the light object's 2D image from a input 2D grid of coordinates and transform this to an array of
        visibilities using a `autoarray.operators.transformer.Transformer` object and therefore a Fourier Transform.

        The input 2D grid may be masked, in which case values outside the mask are not evaluated. This does not impact
        the Fourier transform.

        The grid must be a `Grid2D` objects for certain Fourier transforms to be valid. It therefore cannot be a
        `Grid2DIrregular` objects.

        If the image is all zeros (e.g. because this light object has no light profiles, for example it is a
        `Galaxy` object with only mass profiles) the Fourier transformed is skipped for efficiency and a `Visibilities`
        object with all zeros is returned.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        transformer
            The **PyAutoArray** `Transformer` object describing how the 2D image is Fourier transformed to visiblities
            in the uv-plane.
        """
        image = self.image_2d_from(grid=grid)

        if not np.any(image):
            return aa.Visibilities.zeros(
                shape_slim=(transformer.uv_wavelengths.shape[0],)
            )

        return transformer.visibilities_from(image=image.binned)

    def unmasked_blurred_image_2d_via_psf_from(self, grid, psf):
        """
        Evaluate the light object's 2D image from a input 2D grid of coordinates and convolve it with a PSF, using a
        grid which is not masked.

        Convolving an unmasked 2D image with a PSF requires care, because at the edges of the 2D image the light
        profile values will not be evaluated beyond its edge, even though some of its light will be blurred into these
        edges.

        This function pads the grid first, such that the light profile is evaluated beyond the edge. The function then
        trims the array the image is evaluated on such that it is the original dimensions of the input 2D grid.

        The grid and blurring_grid must be a `Grid2D` objects so the evaluated image can be mapped to a uniform 2D
        array and binned up for convolution. They therefore cannot be `Grid2DIrregular` objects.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        psf
            The PSF the light object 2D image is convolved with.
        """
        padded_grid = grid.padded_grid_from(kernel_shape_native=psf.shape_native)

        padded_image = self.image_2d_from(grid=padded_grid)

        return padded_grid.mask.unmasked_blurred_array_from(
            padded_array=padded_image, psf=psf, image_shape=grid.mask.shape
        )

    def add_functions(self, obj):
        """
        This passes all functions of the `_calc_image` property to the `LightProfile`.

        This means that instead of having to call a function using the full path:

        `light_profile._calc_image.blurred_image_2d_via_psf_from`

        We can simply call it using the path:

        `light_profile.blurred_image_2d_via_psf_from`
        """
        obj.blurred_image_2d_via_psf_from = self.blurred_image_2d_via_psf_from
        obj.blurred_image_2d_via_convolver_from = (
            self.blurred_image_2d_via_convolver_from
        )
        obj.profile_visibilities_via_transformer_from = (
            self.profile_visibilities_via_transformer_from
        )
        obj.unmasked_blurred_image_2d_via_psf_from = (
            self.unmasked_blurred_image_2d_via_psf_from
        )
