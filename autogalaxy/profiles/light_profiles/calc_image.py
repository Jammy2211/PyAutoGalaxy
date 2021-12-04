from typing import Callable, Union

import autoarray as aa


class CalcImage:
    def __init__(self, image_2d_from: Callable):

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

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        transformer
            The **PyAutoArray** `Transformer` object describing how the 2D image is Fourier transformed to visiblities
            in the uv-plane.
        """
        image = self.image_2d_from(grid=grid)

        return transformer.visibilities_from(image=image.binned)
