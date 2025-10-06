from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING, Dict, List, Optional

from autoarray import Array2D

if TYPE_CHECKING:
    from autogalaxy.galaxy.galaxy import Galaxy

import autoarray as aa


class OperateImage:
    """
    Packages methods which operate on the 2D image returned from the `image_2d_from` function of a light object
    (e.g. a `LightProfile`, `Galaxy`).

    The majority of methods apply data operators to the 2D image which perform tasks such as a 2D convolution or
    Fourier transform.

    The methods in `OperateImage` are inherited by light objects to provide a concise API.
    """

    def image_2d_from(
        self, grid: aa.Grid2D, operated_only: Optional[bool] = None
    ) -> aa.Array2D:
        raise NotImplementedError

    def has(self, cls) -> bool:
        raise NotImplementedError

    def _blurred_image_2d_from(
        self,
        image_2d: aa.Array2D,
        blurring_image_2d: aa.Array2D,
        psf: aa.Kernel2D,
    ) -> aa.Array2D:

        values = psf.convolve_image(
            image=image_2d,
            blurring_image=blurring_image_2d,
        )
        return Array2D(values=values, mask=image_2d.mask)

    def blurred_image_2d_from(
        self,
        grid: aa.Grid2D,
        blurring_grid: aa.Grid2D,
        psf: aa.Kernel2D = None,
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
        from autogalaxy.profiles.light.operated import (
            LightProfileOperated,
        )

        image_2d_not_operated = self.image_2d_from(grid=grid, operated_only=False)
        blurring_image_2d_not_operated = self.image_2d_from(
            grid=blurring_grid, operated_only=False
        )

        blurred_image_2d = self._blurred_image_2d_from(
            image_2d=image_2d_not_operated,
            blurring_image_2d=blurring_image_2d_not_operated,
            psf=psf,
        )

        if self.has(cls=LightProfileOperated):
            image_2d_operated = self.image_2d_from(grid=grid, operated_only=True)
            return blurred_image_2d + image_2d_operated

        return blurred_image_2d

    def padded_image_2d_from(self, grid, psf_shape_2d):
        """
        Evaluate the light object's 2D image from a input 2D grid of padded coordinates, where this padding is
        sufficient to encapsulate all surrounding pixels that will blur light into the original image given the
        2D shape of the PSF's kernel.

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

    def unmasked_blurred_image_2d_from(self, grid, psf):
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

        padded_image_2d_not_operated = self.image_2d_from(
            grid=padded_grid, operated_only=False
        )

        padded_image_2d = padded_grid.mask.unmasked_blurred_array_from(
            padded_array=padded_image_2d_not_operated,
            psf=psf,
            image_shape=grid.mask.shape,
        )

        padded_image_2d_operated = self.image_2d_from(
            grid=padded_grid, operated_only=True
        )

        padded_image_2d_operated = padded_grid.mask.trimmed_array_from(
            padded_array=padded_image_2d_operated, image_shape=grid.mask.shape
        )

        return padded_image_2d + padded_image_2d_operated

    def visibilities_from(
        self, grid: aa.Grid2D, transformer: aa.type.Transformer
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

        return jax.lax.cond(
            jnp.any(image_2d.array),
            lambda _: transformer.visibilities_from(image=image_2d),
            lambda _: aa.Visibilities.zeros(
                shape_slim=(transformer.uv_wavelengths.shape[0],)
            ),
            operand=None,
        )


class OperateImageList(OperateImage):
    """
    Packages methods which operate on the list of 2D images returned from the `image_2d_list_from` function of a light
    object which contains multiple light profiles (e.g. a `Galaxy`).

    The majority of methods apply data operators to the list of 2D images which perform tasks such as a 2D convolution
    of Fourier transform.

    The methods in `OperateImageList` are inherited by light objects to provide a concise API.
    """

    def image_2d_list_from(self, grid: aa.Grid2D, operated_only: Optional[bool] = None):
        raise NotImplementedError

    def blurred_image_2d_list_from(
        self,
        grid: aa.Grid2D,
        blurring_grid: aa.Grid2D,
        psf: aa.Kernel2D = None,
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

        image_2d_operated_list = self.image_2d_list_from(grid=grid, operated_only=True)

        image_2d_not_operated_list = self.image_2d_list_from(
            grid=grid, operated_only=False
        )
        blurring_image_2d_not_operated_list = self.image_2d_list_from(
            grid=blurring_grid, operated_only=False
        )

        blurred_image_2d_list = []

        for i in range(len(image_2d_operated_list)):
            image_2d_not_operated = image_2d_not_operated_list[i]
            blurring_image_2d_not_operated = blurring_image_2d_not_operated_list[i]

            blurred_image_2d = self._blurred_image_2d_from(
                image_2d=image_2d_not_operated,
                blurring_image_2d=blurring_image_2d_not_operated,
                psf=psf,
            )

            image_2d_operated = image_2d_operated_list[i]

            blurred_image_2d_list.append(image_2d_operated + blurred_image_2d)

        return blurred_image_2d_list

    def unmasked_blurred_image_2d_list_from(
        self, grid: aa.Grid2D, psf: aa.Kernel2D
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

        padded_image_2d_list = self.image_2d_list_from(grid=padded_grid)

        unmasked_blurred_image_list = []

        for padded_image_2d in padded_image_2d_list:
            unmasked_blurred_array_2d = padded_grid.mask.unmasked_blurred_array_from(
                padded_array=padded_image_2d, psf=psf, image_shape=grid.mask.shape
            )

            unmasked_blurred_image_list.append(unmasked_blurred_array_2d)

        return unmasked_blurred_image_list

    def visibilities_list_from(
        self, grid: aa.Grid2D, transformer: aa.type.Transformer
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
            if not jnp.any(image_2d.array):
                visibilities = aa.Visibilities.zeros(
                    shape_slim=(transformer.uv_wavelengths.shape[0],)
                )
            else:
                visibilities = transformer.visibilities_from(image=image_2d)

            visibilities_list.append(visibilities)

        return visibilities_list


class OperateImageGalaxies(OperateImageList):
    """
    Packages methods which using a list of galaxies returns a dictionary of their 2D images using the function
    `galaxy_image_2d_dict_from` (e.g. galaxies, a `Tracer` in the library **PyAutoLens**).

    The majority of methods apply data operators to the dictionary of 2D images which perform tasks such as a 2D
    convolution of Fourier transform. This retains the keys of the dictionary to maintain information on the galaxies.

    The methods in `OperateImageGalaxies` are inherited by light objects to provide a concise API.
    """

    def galaxy_image_2d_dict_from(
        self, grid: aa.Grid2D, operated_only: Optional[bool] = None
    ) -> Dict[Galaxy, aa.Array2D]:
        raise NotImplementedError

    def galaxy_blurred_image_2d_dict_from(
        self, grid, psf, blurring_grid
    ) -> Dict[Galaxy, aa.Array2D]:
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

        galaxy_image_2d_not_operated_dict = self.galaxy_image_2d_dict_from(
            grid=grid, operated_only=False
        )

        galaxy_blurring_image_2d_not_operated_dict = self.galaxy_image_2d_dict_from(
            grid=blurring_grid, operated_only=False
        )

        galaxy_image_2d_operated_dict = self.galaxy_image_2d_dict_from(
            grid=grid, operated_only=True
        )

        galaxy_blurred_image_2d_dict = {}

        for galaxy_key in galaxy_image_2d_not_operated_dict.keys():
            image_2d_not_operated = galaxy_image_2d_not_operated_dict[galaxy_key]
            blurring_image_2d_not_operated = galaxy_blurring_image_2d_not_operated_dict[
                galaxy_key
            ]

            blurred_image_2d = psf.convolve_image(
                image=image_2d_not_operated,
                blurring_image=blurring_image_2d_not_operated,
            )

            image_2d_operated = galaxy_image_2d_operated_dict[galaxy_key]

            galaxy_blurred_image_2d_dict[galaxy_key] = (
                image_2d_operated + blurred_image_2d
            )

        return galaxy_blurred_image_2d_dict

    def galaxy_visibilities_dict_from(
        self, grid, transformer
    ) -> Dict[Galaxy, aa.Visibilities]:
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

            if not jnp.any(image_2d.array):
                visibilities = aa.Visibilities.zeros(
                    shape_slim=(transformer.uv_wavelengths.shape[0],)
                )

            else:
                visibilities = transformer.visibilities_from(image=image_2d)

            galaxy_visibilities_dict[galaxy_key] = visibilities

        return galaxy_visibilities_dict
