import logging
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

import autoarray as aa
from autoarray.dataset.abstract.settings import AbstractSettingsDataset
from autoarray.dataset.abstract.dataset import AbstractDataset

logger = logging.getLogger(__name__)


class SettingsQuantity(AbstractSettingsDataset):
    def __init__(
        self,
        grid_class=aa.Grid2D,
        sub_size: int = 1,
        fractional_accuracy: float = 0.9999,
        relative_accuracy: Optional[float] = None,
        sub_steps: List[int] = None,
    ):
        """
        The lens dataset is the collection of data_type (image, noise-map, PSF), a mask, grid, convolver
        and other utilities that are used for modeling and fitting an image of a strong lens.

        Whilst the image, noise-map, etc. are loaded in 2D, the lens dataset creates reduced 1D arrays of each
        for lens calculations.

        Parameters
        ----------
        grid_class : ag.Grid2D
            The type of grid used to create the image from the `Galaxy` and `Plane`. The options are `Grid2D`,
            and `Grid2DIterate` (see the `Grid2D` documentation for a description of these options).
        grid_pixelization_class : ag.Grid2D
            The type of grid used to create the grid that maps the `LEq` source pixels to the data's image-pixels.
            The options are `Grid2D` and `Grid2DIterate`.
            (see the `Grid2D` documentation for a description of these options).
        sub_size
            If the grid and / or grid_pixelization use a `Grid2D`, this sets the sub-size used by the `Grid2D`.
        fractional_accuracy
            If the grid and / or grid_pixelization use a `Grid2DIterate`, this sets the fractional accuracy it
            uses when evaluating functions, where the fraction accuracy is the ratio of the values computed using
            two grids at a higher and lower sub-grid size.
        relative_accuracy
            If the grid and / or grid_pixelization use a `Grid2DIterate`, this sets the relative accuracy it
            uses when evaluating functions, where the relative accuracy is the absolute difference of the values
            computed using two grids at a higher and lower sub-grid size.
        sub_steps : [int]
            If the grid and / or grid_pixelization use a `Grid2DIterate`, this sets the steps the sub-size is increased by
            to meet the fractional accuracy when evaluating functions.
        psf_shape_2d
            The shape of the PSF used for convolving model image generated using analytic light profiles. A smaller
            shape will trim the PSF relative to the input image PSF, giving a faster analysis run-time.
        """

        super().__init__(
            grid_class=grid_class,
            sub_size=sub_size,
            fractional_accuracy=fractional_accuracy,
            relative_accuracy=relative_accuracy,
            sub_steps=sub_steps,
        )


class DatasetQuantity(AbstractDataset):
    def __init__(
        self,
        data: Union[aa.Array2D, aa.VectorYX2D],
        noise_map: Union[aa.Array2D, aa.VectorYX2D],
        settings: SettingsQuantity = SettingsQuantity(),
    ):
        """
        Represents a derived quantity of a light profile, mass profile, galaxy or plane as a dataset that can be fitted
        with a model via a non-linear.

        For example, the `DatasetQuantity` could represent the `convergence` of a mass profile, and this dataset could
        then be fit with a `plane` via the `AnalysisQuantity` class. The benefit of doing this is that the components
        of different models can be fitted to one another and matched for example, the power-law model whose convergence
        best matches the convergence of a dark matter profile could be inferred).

        Parameters
        ----------
        data
            The data of the quantity (e.g. 2D convergence, 2D potential, 2D deflections) that is fitted.
        noise_map
            The 2D noise map of the quantity's data, which is often chosen in an arbitrary way.
        settings
            Controls settings of how the dataset is set up (e.g. the types of grids used to perform calculations).
        """

        if data.shape != noise_map.shape:
            if data.shape[0:-1] == noise_map.shape[0:]:
                noise_map = aa.VectorYX2D.no_mask(
                    values=np.stack((noise_map, noise_map), axis=-1),
                    pixel_scales=data.pixel_scales,
                    shape_native=data.shape_native,
                    sub_size=data.sub_size,
                    origin=data.origin,
                )

        self.unmasked = None

        super().__init__(data=data, noise_map=noise_map, settings=settings)

    @classmethod
    def via_signal_to_noise_map(
        cls,
        data: Union[aa.Array2D, aa.VectorYX2D],
        signal_to_noise_map: Union[aa.Array2D],
        settings: SettingsQuantity = SettingsQuantity(),
    ):
        """
        Represents a derived quantity of a light profile, mass profile, galaxy or plane as a dataset that can be fitted
        with a model via a non-linear (see `DatasetQuantity.__init__`).

        This classmethod takes as input a signal-to-noise map, as opposed to the noise map used in the `__init__`
        constructor. The noise-map is then derived from this signal-to-noise map, such that this is the signal to
        noise of the `DatasetQuantity` that is returned.

        Parameters
        ----------
        data
            The data of the quantity (e.g. 2D convergence, 2D potential, 2D deflections) that is fitted.
        signal_to_noise_map
            The 2D signal to noise map of the quantity's data.
        settings
            Controls settings of how the dataset is set up (e.g. the types of grids used to perform calculations).
        """
        try:
            noise_map = data / signal_to_noise_map
        except ValueError:
            noise_map = aa.VectorYX2D.zeros(
                shape_native=data.shape_native, pixel_scales=data.pixel_scales
            )
            noise_map = noise_map.apply_mask(mask=data.mask)

            signal_to_noise_map[signal_to_noise_map < 1e-8] = 1e-8

            noise_map[:, 0] = np.abs(data.slim[:, 0]) / signal_to_noise_map
            noise_map[:, 1] = np.abs(data.slim[:, 1]) / signal_to_noise_map

        return DatasetQuantity(data=data, noise_map=noise_map, settings=settings)

    @property
    def y(self) -> "DatasetQuantity":
        """
        If the `DatasetQuantity` contains a `VectorYX2D` as its data, this property returns a new `DatasetQuantity`
        with just the y-values of the vectors as the data, alongside the noise-map.

        This is primarily used for visualizing a fit to the `DatasetQuantity` containing vectors, as it allows one to
        reuse tools which visualize `Array2D` objects.
        """
        if isinstance(self.data, aa.VectorYX2D):
            return DatasetQuantity(
                data=self.data.y, noise_map=self.noise_map.y, settings=self.settings
            )

    @property
    def x(self) -> "DatasetQuantity":
        """
        If the `DatasetQuantity` contains a `VectorYX2D` as its data, this property returns a new `DatasetQuantity`
        with just the x-values of the vectors as the data, alongside the noise-map.

        This is primarily used for visualizing a fit to the `DatasetQuantity` containing vectors, as it allows one to
        reuse tools which visualize `Array2D` objects.
        """
        if isinstance(self.data, aa.VectorYX2D):
            return DatasetQuantity(
                data=self.data.x, noise_map=self.noise_map.x, settings=self.settings
            )

    def apply_mask(self, mask: aa.Mask2D) -> "DatasetQuantity":
        """
        Apply a mask to the quantity dataset, whereby the mask is applied to the data and noise-map one-by-one.

        The original unmasked qunatity dataset is stored as the `self.unmasked` attribute. This is used to ensure that
        if the `apply_mask` function is called multiple times, every mask is always applied to the original unmasked
        imaging dataset.

        Parameters
        ----------
        mask
            The 2D mask that is applied to the image.
        """
        if self.data.mask.is_all_false:
            unmasked_dataset = self
        else:
            unmasked_dataset = self.unmasked

        data = self.data.apply_mask(mask=mask.derive_mask.sub_1)
        noise_map = self.noise_map.apply_mask(mask=mask.derive_mask.sub_1)

        dataset = DatasetQuantity(
            data=data, noise_map=noise_map, settings=self.settings
        )

        dataset.unmasked = unmasked_dataset

        logger.info(
            f"IMAGING - Data masked, contains a total of {mask.pixels_in_mask} image-pixels"
        )

        return dataset

    def apply_settings(self, settings: SettingsQuantity) -> "DatasetQuantity":
        """
        Returns a new instance of the quantity dataset with the input `SettingsQuantity` applied to them.

        This can be used to update settings like the types of grids associated with the dataset that are used
        to perform calculations.

        Parameters
        ----------
        settings
            The settings for the quantity dataset which control things like the grids used for calculations.
        """
        return DatasetQuantity(
            data=self.data, noise_map=self.noise_map, settings=settings
        )

    @property
    def shape_native(self):
        return self.data.shape_native

    @property
    def pixel_scales(self):
        return self.data.pixel_scales

    def output_to_fits(
        self,
        data_path: Union[Path, str],
        noise_map_path: Optional[Union[Path, str]] = None,
        overwrite: bool = False,
    ):
        """
        Output a quantity dataset to multiple .fits file.

        For each attribute of the imaging data (e.g. `data`, `noise_map`) the path to
        the .fits can be specified, with `hdu=0` assumed automatically.

        If the `data` has been masked, the masked data is output to .fits files. A mask can be separately output to
        a file `mask.fits` via the `Mask` objects `output_to_fits` method.

        Parameters
        ----------
        data_path
            The path to the data .fits file where the image data is output (e.g. '/path/to/data.fits').
        noise_map_path
            The path to the noise_map .fits where the noise_map is output (e.g. '/path/to/noise_map.fits').
        overwrite
            If `True`, the .fits files are overwritten if they already exist, if `False` they are not and an
            exception is raised.
        """
        self.data.output_to_fits(file_path=data_path, overwrite=overwrite)

        if self.noise_map is not None and noise_map_path is not None:
            self.noise_map.output_to_fits(file_path=noise_map_path, overwrite=overwrite)
