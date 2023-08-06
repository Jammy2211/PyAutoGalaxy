from typing import Optional

import autofit as af


class SetupAdapt:
    def __init__(
        self,
        search_pix_cls: Optional[af.NonLinearSearch] = None,
        search_pix_dict: Optional[dict] = None,
        mesh_pixels_fixed: Optional[int] = None,
    ):
        """
        The adapt setup of a pipeline, which controls how adaptive-features in PyAutoGalaxy template pipelines run,
        for example controlling whether galaxies are used to scale the noise and the non-linear searches used
        in these searchs.

        Users can write their own pipelines which do not use or require the *SetupAdapt* class.

        Parameters
        ----------
        search_pix_cls
            The non-linear search used by every adapt model-fit search.
        search_pix_dict
            The dictionary of search options for the adapt inversion model-fit searches.
        """

        self.search_pix_cls = search_pix_cls or af.Nautilus
        self.search_pix_dict = search_pix_dict or {
            "n_live": 75,
        }

        self.mesh_pixels_fixed = mesh_pixels_fixed
