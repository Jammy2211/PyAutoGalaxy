from typing import Optional

import autofit as af

from autogalaxy.legacy.hyper_data import HyperImageSky
from autogalaxy.legacy.hyper_data import HyperBackgroundNoise


class SetupHyper:
    def __init__(
        self,
        search_pix_cls: Optional[af.NonLinearSearch] = None,
        search_pix_dict: Optional[dict] = None,
        mesh_pixels_fixed: Optional[int] = None,
    ):
        """
        The hyper setup of a pipeline, which controls how hyper-features in PyAutoGalaxy template pipelines run,
        for example controlling whether hyper galaxies are used to scale the noise and the non-linear searches used
        in these searchs.

        Users can write their own pipelines which do not use or require the *SetupHyper* class.

        Parameters
        ----------
        search_pix_cls
            The non-linear search used by every hyper model-fit search.
        search_pix_dict
            The dictionary of search options for the hyper inversion model-fit searches.
        """

        self.search_pix_cls = search_pix_cls or af.DynestyStatic
        self.search_pix_dict = search_pix_dict or {
            "nlive": 50,
            "sample": "rwalk",
            "dlogz": 10,
        }

        self.mesh_pixels_fixed = mesh_pixels_fixed
