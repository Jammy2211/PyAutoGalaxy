from __future__ import annotations
import csv
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
from autogalaxy.galaxy.plot.galaxies_plotters import GalaxiesPlotter
from autogalaxy.galaxy.plot.adapt_plotters import AdaptPlotter

from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D


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
        """
        Provides an interface between an output path and all plotter objects.

        This is used to visualize the results of a model-fit, where the `image_path` points to the
        folder where the results of the model-fit are stored on your hard-disk, which is typically the `image` folder
        of a non-linear search.

        The `PlotterInterface` is typically used in the `Analysis` class of a non-linear search to visualize the maximum
        log likelihood model of the model-fit so far.

        The methods of the `PlotterInterface` are called throughout a non-linear search using the `Analysis.Visualizer`
        classes `visualize` method.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml`.

        Parameters
        ----------
        image_path
            The path on the hard-disk to the `image` folder of the non-linear searches results.
        title_prefix
            A string that is added before the title of all figures output by visualization, for example to
            put the name of the dataset and galaxy in the title.
        """
        from pathlib import Path

        self.image_path = Path(image_path)
        self.title_prefix = title_prefix

        os.makedirs(image_path, exist_ok=True)

    @property
    def fmt(self) -> List[str]:
        return conf.instance["visualize"]["plots"]["subplot_format"]

    def mat_plot_1d_from(self) -> MatPlot1D:
        """
        Returns a 1D matplotlib plotting object whose `Output` class uses the `image_path`, such that it outputs
        images to the `image` folder of the non-linear search.

        Returns
        -------
        MatPlot1D
            The 1D matplotlib plotter object.
        """
        return MatPlot1D(
            title=aplt.Title(prefix=self.title_prefix),
            output=aplt.Output(path=self.image_path, format=self.fmt),
        )

    def mat_plot_2d_from(self, quick_update: bool = False) -> MatPlot2D:
        """
        Returns a 2D matplotlib plotting object whose `Output` class uses the `image_path`, such that it outputs
        images to the `image` folder of the non-linear search.

        Returns
        -------
        MatPlot2D
            The 2D matplotlib plotter object.
        """
        return MatPlot2D(
            title=aplt.Title(prefix=self.title_prefix),
            output=aplt.Output(path=self.image_path, format=self.fmt),
            quick_update=quick_update,
        )

    def galaxies(
        self,
        galaxies: List[Galaxy],
        grid: aa.type.Grid2DLike,
    ):
        """
        Visualizes a list of galaxies.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the
        `image_path` points to the search's results folder and this function visualizes the maximum log likelihood
        galaxies inferred by the search so far.

        Visualization includes subplots of the individual images of attributes of the galaxies (e.g. its image,
        convergence, deflection angles) and .fits files containing these attributes.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `galaxies` header.

        Parameters
        ----------
        galaxies
            The maximum log likelihood galaxies of the non-linear search.
        grid
            A 2D grid of (y,x) arc-second coordinates used to perform ray-tracing, which is the masked grid tied to
            the dataset.
        """

        galaxies = Galaxies(galaxies=galaxies)

        def should_plot(name):
            return plot_setting(section="galaxies", name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        plotter = GalaxiesPlotter(
            galaxies=galaxies,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_galaxy_images"):
            plotter.subplot_galaxy_images()

        mat_plot_2d = self.mat_plot_2d_from()

        plotter = GalaxiesPlotter(
            galaxies=galaxies,
            grid=grid,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_galaxies"):
            plotter.subplot()

        mat_plot_1d = self.mat_plot_1d_from()

        galaxies_plotter = GalaxiesPlotter(
            galaxies=galaxies,
            grid=grid,
            mat_plot_1d=mat_plot_1d,
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
        """
        Visualizes an `Inversion` object.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        points to the search's results folder and this function visualizes the maximum log likelihood `Inversion`
        inferred by the search so far.

        Visualization includes subplots of individual images of attributes of the dataset (e.g. the reconstructed image,
        the reconstruction) and .fits file of attributes.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `inversion` header.

        Parameters
        ----------
        inversion
            The inversion used to fit the dataset whose attributes are visualized.
        """

        def should_plot(name):
            return plot_setting(section="inversion", name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        inversion_plotter = aplt.InversionPlotter(
            inversion=inversion,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_inversion"):
            mapper_list = inversion.cls_list_from(cls=aa.AbstractMapper)

            for i in range(len(mapper_list)):
                suffix = "" if len(mapper_list) == 1 else f"_{i}"

                inversion_plotter.subplot_of_mapper(
                    mapper_index=i, auto_filename=f"subplot_inversion{suffix}"
                )

        if should_plot("csv_reconstruction"):
            mapper_list = inversion.cls_list_from(cls=aa.AbstractMapper)

            for i, mapper in enumerate(mapper_list):
                y = mapper.mapper_grids.source_plane_mesh_grid[:, 0]
                x = mapper.mapper_grids.source_plane_mesh_grid[:, 1]
                reconstruction = inversion.reconstruction_dict[mapper]
                noise_map = inversion.reconstruction_noise_map_dict[mapper]

                with open(
                    self.image_path / f"source_plane_reconstruction_{i}.csv",
                    mode="w",
                    newline="",
                ) as file:
                    writer = csv.writer(file)
                    writer.writerow(["y", "x", "reconstruction", "noise_map"])  # header

                    for i in range(len(x)):
                        writer.writerow(
                            [
                                float(y[i]),
                                float(x[i]),
                                float(reconstruction[i]),
                                float(noise_map[i]),
                            ]
                        )

    def adapt_images(
        self,
        adapt_images: AdaptImages,
    ):
        """
        Visualizes the adapt images used by a model-fit for adaptive pixelization mesh's and regularization.

        Images are output to the `image` folder of the `image_path`. When used with a non-linear search the `image_path`
        is the output folder of the non-linear search.

        Visualization includes a subplot image of all galaxy images on the same figure.

        The images output by the `PlotterInterface` are customized using the file `config/visualize/plots.yaml` under
        the `adapt` header.

        Parameters
        ----------
        adapt_images
            The adapt images (e.g. overall model image, individual galaxy images).
        """

        def should_plot(name):
            return plot_setting(section="adapt", name=name)

        mat_plot_2d = self.mat_plot_2d_from()

        adapt_plotter = AdaptPlotter(
            mat_plot_2d=mat_plot_2d,
        )

        if adapt_images.galaxy_name_image_dict is not None:

            if should_plot("subplot_adapt_images"):
                adapt_plotter.subplot_adapt_images(
                    adapt_galaxy_name_image_dict=adapt_images.galaxy_name_image_dict
                )

        if should_plot("fits_adapt_images"):

            if adapt_images.galaxy_name_image_dict is not None:

                image_list = [
                    adapt_images.galaxy_name_image_dict[name].native_for_fits
                    for name in adapt_images.galaxy_name_image_dict.keys()
                ]

                hdu_list = hdu_list_for_output_from(
                    values_list=[
                        image_list[0].mask.astype("float"),
                    ]
                    + image_list,
                    ext_name_list=["mask"]
                    + list(adapt_images.galaxy_name_image_dict.keys()),
                    header_dict=adapt_images.mask.header_dict,
                )

                hdu_list.writeto(self.image_path / "adapt_images.fits", overwrite=True)

            if adapt_images.galaxy_name_image_plane_mesh_grid_dict is not None:

                image_plane_mesh_grid_list = [
                    adapt_images.galaxy_name_image_plane_mesh_grid_dict[name].native
                    for name in adapt_images.galaxy_name_image_plane_mesh_grid_dict.keys()
                ]

                print(list(adapt_images.galaxy_name_image_plane_mesh_grid_dict.keys()))

                hdu_list = hdu_list_for_output_from(
                    values_list=[np.array([1])] + image_plane_mesh_grid_list,
                    ext_name_list=[""]
                    + list(adapt_images.galaxy_name_image_plane_mesh_grid_dict.keys()),
                )

                hdu_list.writeto(
                    self.image_path / "adapt_image_plane_mesh_grids.fits",
                    overwrite=True,
                )
