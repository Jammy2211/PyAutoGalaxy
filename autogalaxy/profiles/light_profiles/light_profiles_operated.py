import numpy as np

import autoarray as aa

from autogalaxy.profiles.light_profiles import light_profiles as lp
from autogalaxy.profiles.light_profiles import light_profiles_linear as lp_linear

from autogalaxy import exc

class LightProfileOperated:

    pass


class EllGaussian(lp.EllGaussian, LightProfileOperated):

    pass

    def image_2d_not_operated_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        raise exc.ProfileException(
            "Cannot call `image_2d_not_operated_from() method for a LightProfileOperated object."
        )

class EllGaussianLinear(lp_linear.EllGaussian, LightProfileOperated):

    pass

    def image_2d_not_operated_from(self, grid: aa.type.Grid2DLike) -> np.ndarray:
        raise exc.ProfileException(
            "Cannot call `image_2d_not_operated_from() method for a LightProfileOperated object."
        )
