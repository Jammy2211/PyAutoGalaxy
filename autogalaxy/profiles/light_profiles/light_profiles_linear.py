import numpy as np
from typing import Dict, Optional, Tuple, Union

from autoconf import cached_property
import autoarray as aa

from autogalaxy.profiles.light_profiles import light_profiles as lp


class LightProfileLinear(lp.LightProfile):

    pass


class LightProfileLinearObjFunc(aa.LinearObjFunc):
    def __init__(
        self,
        grid: aa.type.Grid1D2DLike,
        blurring_grid: aa.type.Grid1D2DLike,
        convolver: Optional[aa.Convolver],
        light_profile: LightProfileLinear,
        profiling_dict: Optional[Dict] = None,
    ):

        super().__init__(grid=grid, profiling_dict=profiling_dict)

        self.blurring_grid = blurring_grid
        self.convolver = convolver
        self.light_profile = light_profile

    @property
    def mapping_matrix(self) -> np.ndarray:
        return self.light_profile.image_2d_from(grid=self.grid).binned.slim[:, None]

    @cached_property
    def blurred_mapping_matrix_override(self) -> Optional[np.ndarray]:
        """
        The `LinearEqn` object takes the `mapping_matrix` of each linear object and combines it with the `Convolver`
        operator to perform a 2D convolution and compute the `blurred_mapping_matrix`.

        If this property is overwritten this operation is not performed, with the `blurred_mapping_matrix` output this
        property automatically used instead.

        This is used for a linear light profile because the in-built mapping matrix convolution does not account for
        how light profile images have flux outside the masked region which is blurred into the masked region. This
        flux is outside the region that defines the `mapping_matrix` and thus this override is required to properly
        incorporate it.

        Returns
        -------
        A blurred mapping matrix of dimensions (total_mask_pixels, 1) which overrides the mapping matrix calculations
        performed in the linear equation solvers.
        """

        image_2d = self.light_profile.image_2d_from(grid=self.grid)

        blurring_image_2d = self.light_profile.image_2d_from(grid=self.blurring_grid)

        return self.convolver.convolve_image(
            image=image_2d, blurring_image=blurring_image_2d
        )[:, None]


class EllSersic(lp.EllSersic, LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
    ):
        """
        The elliptical Sersic light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            intensity=1.0,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
        )


class EllGaussian(lp.EllGaussian, LightProfileLinear):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        sigma: float = 0.01,
    ):
        """
        The elliptical Gaussian light profile.

        See `autogalaxy.profiles.light_profiles.light_profiles.LightProfile` for a description of light profile objects.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        intensity
            Overall intensity normalisation of the light profile (units are dimensionless and derived from the data
            the light profile's image is compared too, which is expected to be electrons per second).
        sigma
            The sigma value of the Gaussian, corresponding to ~ 1 / sqrt(2 log(2)) the full width half maximum.
        """

        super().__init__(
            centre=centre, elliptical_comps=elliptical_comps, intensity=1.0, sigma=sigma
        )
