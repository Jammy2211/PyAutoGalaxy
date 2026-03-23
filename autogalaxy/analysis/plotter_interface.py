from __future__ import annotations
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from autoconf import conf
from autoconf.fitsable import hdu_list_for_output_from

import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.analysis.adapt_images.adapt_images import AdaptImages
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.galaxies import Galaxies
from autogalaxy.galaxy.plot import galaxies_plots
from autogalaxy.galaxy.plot import adapt_plots


def setting(section: Union[List[str], str], name: str):
    if isinstance(section, str):
        return conf.instance["visualize"]["plots"][section][name]

    for sect in reversed(section):
        try:
            return conf.instance["visualize"]["plots"][sect][name]
        except KeyError:
            continue

    return conf.instance["visualize"]["plots"][section[0]][name]


def plot_setting(section: Union[List[str], str], name: str) -> bool:
    return setting(section, name)


class PlotterInterface:
    def __init__(self, image_path: Union[Path, str], title_prefix: str = None):
        from pathlib import Path

        self.image_path = Path(image_path)
        self.title_prefix = title_prefix

        os.makedirs(image_path, exist_ok=True)

    @property
    def fmt(self) -> List[str]:
        return conf.instance["visualize"]["plots"]["subplot_format"]

    def output_from(self) -> aplt.Output:
        return aplt.Output(path=self.image_path, format=self.fmt)

    def galaxies(
        self,
        galaxies: List[Galaxy],
        grid: aa.type.Grid2DLike,
    ):
        galaxies = Galaxies(galaxies=galaxies)

        def should_plot(name):
            return plot_setting(section="galaxies", name=name)

        if should_plot("subplot_galaxy_images"):
            galaxies_plots.subplot_galaxy_images(
                galaxies=galaxies,
                grid=grid,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        if should_plot("subplot_galaxies"):
            galaxies_plots.subplot_galaxies(
                galaxies=galaxies,
                grid=grid,
                output_path=self.image_path,
                output_format=self.fmt,
            )

        if should_plot("fits_galaxy_images"):
            image_list = [
                galaxy.image_2d_from(grid=grid).native_for_fits for galaxy in galaxies
            ]

            hdu_list = hdu_list_for_output_from(
                values_list=[image_list[0].mask.astype("float")] + image_list,
                ext_name_list=["mask"] + [f"galaxy_{i}" for i in range(len(galaxies))],
                header_dict=grid.mask.header_dict,
            )

            hdu_list.writeto(self.image_path / "galaxy_images.fits", overwrite=True)

    def inversion(self, inversion: aa.Inversion):
        def should_plot(name):
            return plot_setting(section="inversion", name=name)

        output = self.output_from()

        if should_plot("subplot_inversion"):
            from autogalaxy.plot.plot_utils import _save_subplot

            mapper_list = inversion.cls_list_from(cls=aa.Mapper)

            for i, mapper in enumerate(mapper_list):
                suffix = "" if len(mapper_list) == 1 else f"_{i}"
                reconstruction = inversion.reconstruction_dict[mapper]
                grid = np.array(mapper.source_plane_mesh_grid)

                fig, ax = plt.subplots(1, 1, figsize=(7, 7))
                sc = ax.scatter(grid[:, 1], grid[:, 0], c=reconstruction, s=10)
                plt.colorbar(sc, ax=ax)
                ax.set_title("Reconstruction")
                ax.set_aspect("equal")
                _save_subplot(fig, output.path, f"subplot_inversion{suffix}", output.format_list[0] if output.format_list else "png")

        if should_plot("csv_reconstruction"):
            mapper_list = inversion.cls_list_from(cls=aa.Mapper)

            for i, mapper in enumerate(mapper_list):
                y = mapper.source_plane_mesh_grid[:, 0]
                x = mapper.source_plane_mesh_grid[:, 1]
                reconstruction = inversion.reconstruction_dict[mapper]
                noise_map = inversion.reconstruction_noise_map_dict[mapper]

                with open(
                    self.image_path / f"source_plane_reconstruction_{i}.csv",
                    mode="w",
                    newline="",
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(["y", "x", "reconstruction", "noise_map"])

                    for i in range(len(x)):
                        writer.writerow(
                            [
                                float(y[i]),
                                float(x[i]),
                                float(reconstruction[i]),
                                float(noise_map[i]),
                            ]
                        )

    def adapt_images(self, adapt_images: AdaptImages):
        def should_plot(name):
            return plot_setting(section="adapt", name=name)

        if adapt_images.galaxy_name_image_dict is not None:
            if should_plot("subplot_adapt_images"):
                adapt_plots.subplot_adapt_images(
                    adapt_galaxy_name_image_dict=adapt_images.galaxy_name_image_dict,
                    output_path=self.image_path,
                    output_format=self.fmt,
                )

        if should_plot("fits_adapt_images"):
            if adapt_images.galaxy_name_image_dict is not None:
                image_list = [
                    adapt_images.galaxy_name_image_dict[name].native_for_fits
                    for name in adapt_images.galaxy_name_image_dict.keys()
                ]

                hdu_list = hdu_list_for_output_from(
                    values_list=[image_list[0].mask.astype("float")] + image_list,
                    ext_name_list=["mask"] + list(adapt_images.galaxy_name_image_dict.keys()),
                    header_dict=adapt_images.mask.header_dict,
                )

                hdu_list.writeto(self.image_path / "adapt_images.fits", overwrite=True)

            if adapt_images.galaxy_name_image_plane_mesh_grid_dict is not None:
                image_plane_mesh_grid_list = [
                    adapt_images.galaxy_name_image_plane_mesh_grid_dict[name].native
                    for name in adapt_images.galaxy_name_image_plane_mesh_grid_dict.keys()
                ]

                hdu_list = hdu_list_for_output_from(
                    values_list=[np.array([1])] + image_plane_mesh_grid_list,
                    ext_name_list=[""]
                    + list(adapt_images.galaxy_name_image_plane_mesh_grid_dict.keys()),
                )

                hdu_list.writeto(
                    self.image_path / "adapt_image_plane_mesh_grids.fits",
                    overwrite=True,
                )
