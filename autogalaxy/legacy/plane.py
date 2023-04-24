import numpy as np
from typing import Dict, List, Optional

import autoarray as aa

from autogalaxy.legacy.hyper import HyperGalaxy
from autogalaxy.plane.plane import Plane as PlaneBase


class Plane(PlaneBase):
    def __init__(
        self,
        galaxies,
        redshift: Optional[float] = None,
        profiling_dict: Optional[Dict] = None,
    ):
        """
        A plane of galaxies where all galaxies are at the same redshift.

        Parameters
        ----------
        redshift or None
            The redshift of the plane.
        galaxies : [Galaxy]
            The list of galaxies in this plane.
        """

        super().__init__(
            redshift=redshift, galaxies=galaxies, profiling_dict=profiling_dict
        )

    def hyper_noise_map_from(self, noise_map) -> aa.Array2D:
        hyper_noise_maps = self.hyper_noise_map_list_from(noise_map=noise_map)
        return sum(hyper_noise_maps)

    def hyper_noise_map_list_from(self, noise_map) -> List[aa.Array2D]:
        """
        For a contribution map and noise-map, use the model hyper_galaxy galaxies to compute a hyper noise-map.

        Parameters
        ----------
        noise_map : imaging.NoiseMap or ndarray
            An arrays describing the RMS standard deviation error in each pixel, preferably in units of electrons per
            second.
        """
        hyper_noise_map_list = []

        for galaxy in self.galaxies:
            if galaxy.has(cls=HyperGalaxy):
                contribution_map = galaxy.hyper_galaxy.contribution_map_from(
                    adapt_model_image=galaxy.adapt_model_image,
                    adapt_galaxy_image=galaxy.adapt_galaxy_image,
                )

                hyper_noise_map = galaxy.hyper_galaxy.hyper_noise_map_from(
                    contribution_map=contribution_map,
                    noise_map=noise_map,
                )

                hyper_noise_map_list.append(hyper_noise_map)

            else:
                hyper_noise_map = aa.Array2D(
                    values=np.zeros(noise_map.mask.derive_mask.sub_1.pixels_in_mask),
                    mask=noise_map.mask.derive_mask.sub_1,
                )

                hyper_noise_map_list.append(hyper_noise_map)

        return hyper_noise_map_list

    @property
    def contribution_map(self) -> Optional[aa.Array2D]:
        contribution_map_list = self.contribution_map_list

        contribution_map_list = [i for i in contribution_map_list if i is not None]

        if contribution_map_list:
            return sum(contribution_map_list)
        else:
            return None

    @property
    def contribution_map_list(self) -> List[aa.Array2D]:
        contribution_map_list = []

        for galaxy in self.galaxies:
            if galaxy.hyper_galaxy is not None:
                contribution_map_list.append(galaxy.contribution_map)

            else:
                contribution_map_list.append(None)

        return contribution_map_list
