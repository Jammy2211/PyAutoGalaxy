import numpy as np
from typing import Dict, List, Union

import autoarray as aa


class OperateImage:
    """
    Packages methods which operate on the 2D image returned from the `image_2d_from` function of a light object
    (e.g. a `LightProfile`, `Galaxy`, `Plane`).

    The majority of methods apply data operators to the 2D image which perform tasks such as a 2D convolution or
    Fourier transform.

    The methods in `OperateImage` are inherited by light objects to provide a concise API.
    """

    def image_2d_from(self, grid: Union[aa.Grid2D, aa.Grid2DIterate]) -> aa.Array2D:
        raise NotImplementedError

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

        image_2d = self.image_2d_from(grid=grid)
        blurring_image_2d = self.image_2d_from(grid=blurring_grid)

        return psf.convolved_array_with_mask_from(
            array=image_2d.binned.native + blurring_image_2d.binned.native,
            mask=grid.mask,
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

        image_2d = self.image_2d_from(grid=grid)
        blurring_image_2d = self.image_2d_from(grid=blurring_grid)

        return convolver.convolve_image(
            image=image_2d.binned, blurring_image=blurring_image_2d.binned
        )

    def padded_image_2d_from(self, grid, psf_shape_2d):
        """
        Evaluate the light object's 2D image from a input 2D grid of padded coordinates, where this padding is
        sufficient to encapsulate all surrounding pixels that will blur light into the original image given the
        2D shape of the PSF's kernel..

        Convolving an unmasked 2D image with a PSF requires care, because at the edges of the 2D image the light
        profile values will not be evaluated beyond its edge, even though some of its light will be blurred into these
        edges.

        This function creates the padded image, such that the light profile is evaluated beyond the edge. The
        array will still require trimming to remove these additional pixels after convolution is performed.

        Parameters
        ----------
        grid
            The 2D (y,x) coordinates of the (masked) grid, in its original geometric reference frame.
        psf_shape_2d
            The 2D shape of the PSF the light object 2D image is convolved with.
        """
        padded_grid = grid.padded_grid_from(kernel_shape_native=psf_shape_2d)

        return self.image_2d_from(grid=padded_grid)

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

    def visibilities_via_transformer_from(
        self, grid: Union[aa.Grid2D, aa.Grid2DIterate], transformer: aa.type.Transformer
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

        image_2d = self.image_2d_from(grid=grid)

        if not np.any(image_2d):
            return aa.Visibilities.zeros(
                shape_slim=(transformer.uv_wavelengths.shape[0],)
            )

        return transformer.visibilities_from(image=image_2d.binned)


class OperateImageList(OperateImage):
    """
    Packages methods which operate on the list of 2D images returned from the `image_2d_list_from` function of a light
    object which contains multiple light profiles (e.g. a `Galaxy`, `Plane`).

    The majority of methods apply data operators to the list of 2D images which perform tasks such as a 2D convolution
    of Fourier transform.

    The methods in `OperateImageList` are inherited by light objects to provide a concise API.
    """

    def image_2d_list_from(self, grid: Union[aa.Grid2D, aa.Grid2DIterate]):
        raise NotImplementedError

    def blurred_image_2d_list_via_psf_from(
        self,
        grid: Union[aa.Grid2D, aa.Grid2DIterate],
        psf,
        blurring_grid: Union[aa.Grid2D, aa.Grid2DIterate],
    ) -> List[aa.Array2D]:
        """
        Evaluate the light object's list of 2D images from a input 2D grid of coordinates and convolve each image with
        a PSF.

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

        image_2d_list = self.image_2d_list_from(grid=grid)
        blurring_image_2d_list = self.image_2d_list_from(grid=blurring_grid)

        blurred_image_2d_list = []

        for image_2d, blurring_image_2d in zip(image_2d_list, blurring_image_2d_list):

            blurred_image_2d = psf.convolved_array_with_mask_from(
                array=image_2d.binned.native + blurring_image_2d.binned.native,
                mask=grid.mask,
            )

            blurred_image_2d_list.append(blurred_image_2d)

        return blurred_image_2d_list

    def blurred_image_2d_list_via_convolver_from(
        self,
        grid: Union[aa.Grid2D, aa.Grid2DIterate],
        convolver: aa.Convolver,
        blurring_grid: Union[aa.Grid2D, aa.Grid2DIterate],
    ) -> List[aa.Array2D]:
        """
        Evaluate the light object's list of 2D images from a input 2D grid of coordinates and convolve each image with
        a PSF, using a `autoarray.operators.convolver.Convolver` object. The `Convolver` object performs the 2D
        convolution operations using 1D NumPy arrays without mapping them to 2D, which is more efficient.

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
        image_2d_list = self.image_2d_list_from(grid=grid)
        blurring_image_2d_list = self.image_2d_list_from(grid=blurring_grid)

        blurred_image_2d_list = []

        for image_2d, blurring_image_2d in zip(image_2d_list, blurring_image_2d_list):

            blurred_image_2d = convolver.convolve_image(
                image=image_2d.binned, blurring_image=blurring_image_2d.binned
            )

            blurred_image_2d_list.append(blurred_image_2d)

        return blurred_image_2d_list

    def unmasked_blurred_image_2d_list_via_psf_from(
        self, grid: Union[aa.Grid2D, aa.Grid2DIterate], psf: aa.Kernel2D
    ) -> List[aa.Array2D]:
        """
        Evaluate the light object's list of 2D images from a input 2D grid of coordinates and convolve it with a PSF,
        using a grid which is not masked.

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

        padded_image_1d_list = self.image_2d_list_from(grid=padded_grid)

        unmasked_blurred_image_list = []

        for padded_image_1d in padded_image_1d_list:

            unmasked_blurred_array_2d = padded_grid.mask.unmasked_blurred_array_from(
                padded_array=padded_image_1d, psf=psf, image_shape=grid.mask.shape
            )

            unmasked_blurred_image_list.append(unmasked_blurred_array_2d)

        return unmasked_blurred_image_list

    def visibilities_list_via_transformer_from(
        self, grid: Union[aa.Grid2D, aa.Grid2DIterate], transformer: aa.type.Transformer
    ) -> List[aa.Array2D]:
        """
        Evaluate the light object's list of 2D image from a input 2D grid of coordinates and transform each image to
        arrays of visibilities using a `autoarray.operators.transformer.Transformer` object and therefore a Fourier
        Transform.

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
        image_2d_list = self.image_2d_list_from(grid=grid)

        visibilities_list = []

        for image_2d in image_2d_list:

            if not np.any(image_2d):
                visibilities = aa.Visibilities.zeros(
                    shape_slim=(transformer.uv_wavelengths.shape[0],)
                )
            else:
                visibilities = transformer.visibilities_from(image=image_2d.binned)

            visibilities_list.append(visibilities)

        return visibilities_list


class OperateImageGalaxies(OperateImageList):
    """
    Packages methods which using a list of galaxies returns a dictionary of their 2D images using the function 
    `galaxy_image_2d_dict_from` (e.g. a `Plane`, a `Tracer` in the library **PyAutoLens**).

    The majority of methods apply data operators to the dictionary of 2D images which perform tasks such as a 2D 
    convolution of Fourier transform. This retains the keys of the dictionary to maintain information on the galaxies.

    The methods in `OperateImageGalaxies` are inherited by light objects to provide a concise API.
    """

    def galaxy_image_2d_dict_from(
        self, grid: Union[aa.Grid2D, aa.Grid2DIterate]
    ) -> Dict:
        raise NotImplementedError

    def galaxy_blurred_image_2d_dict_via_convolver_from(
        self, grid, convolver, blurring_grid
    ) -> Dict:
        """
        Evaluate the light object's dictionary mapping galaixes to their corresponding 2D images and convolve each
        image with a PSF.

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

        galaxy_image_2d_dict = self.galaxy_image_2d_dict_from(grid=grid)
        galaxy_blurring_image_2d_dict = self.galaxy_image_2d_dict_from(
            grid=blurring_grid
        )

        galaxy_blurred_image_2d_dict = {}

        for galaxy_key in galaxy_image_2d_dict.keys():

            image_2d = galaxy_image_2d_dict[galaxy_key]
            blurring_image_2d = galaxy_blurring_image_2d_dict[galaxy_key]

            blurred_image_2d = convolver.convolve_image(
                image=image_2d.binned, blurring_image=blurring_image_2d.binned
            )

            galaxy_blurred_image_2d_dict[galaxy_key] = blurred_image_2d

        return galaxy_blurred_image_2d_dict

    def galaxy_visibilities_dict_via_transformer_from(self, grid, transformer) -> Dict:
        """
        Evaluate the light object's dictionary mapping galaixes to their corresponding 2D images and transform each
        image to arrays of visibilities using a `autoarray.operators.transformer.Transformer` object and therefore a
        Fourier Transform.

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

        galaxy_image_2d_dict = self.galaxy_image_2d_dict_from(grid=grid)

        galaxy_visibilities_dict = {}

        for galaxy_key in galaxy_image_2d_dict.keys():

            image_2d = galaxy_image_2d_dict[galaxy_key]

            if not np.any(image_2d):
                visibilities = aa.Visibilities.zeros(
                    shape_slim=(transformer.uv_wavelengths.shape[0],)
                )

            else:

                visibilities = transformer.visibilities_from(image=image_2d.binned)

            galaxy_visibilities_dict[galaxy_key] = visibilities

        return galaxy_visibilities_dict
