import numpy as np
from functools import wraps
from typing import Optional, Union

import autoarray as aa


def check_operated_only(func):
    """
    Checks if a light profile is a `LightProfileOperated` class and therefore already has had operations like a
    PSF convolution performed.

    This is compared to the `only_operated` input to determine if the image of that light profile is returned, or
    an array of zeros.

    Parameters
    ----------
    func
        A function which checks the light profile class and determines what image is returned.

    Returns
    -------
        A function that returns a 2D image.
    """

    @wraps(func)
    def wrapper(
        obj,
        grid: aa.type.Grid1D2DLike,
        operated_only: Optional[bool] = None,
        *args,
        **kwargs
    ) -> Union[aa.Array2D, np.ndarray]:
        """
        This decorator checks if a light profile is a `LightProfileOperated` class and therefore already has had operations like a
        PSF convolution performed.

        This is compared to the `only_operated` input to determine if the image of that light profile is returned, or
        an array of zeros.

        Parameters
        ----------
        obj
            A light profile with an `image_2d_from` function whose class is inspected to determine if the image is
            operated on.
        grid
            A grid_like object of (y,x) coordinates on which the function values are evaluated.
        operated_only
            By default this is None and the image is returned irrespecive of light profile class (E.g. it does not matter
            if it is already operated or not). If this input is included as a bool, the light profile image is only
            returned if they are or are not already operated.

        Returns
        -------
            The 2D image, which is customized depending on whether it has been operated on.
        """

        from autogalaxy.profiles.light.operated import (
            LightProfileOperated,
        )

        if operated_only is None:
            return func(obj, grid, operated_only, *args, **kwargs)
        elif operated_only:
            if isinstance(obj, LightProfileOperated):
                return func(obj, grid, operated_only, *args, **kwargs)
            return np.zeros((grid.shape[0],))
        if not isinstance(obj, LightProfileOperated):
            return func(obj, grid, operated_only, *args, **kwargs)
        return np.zeros((grid.shape[0],))

    return wrapper
