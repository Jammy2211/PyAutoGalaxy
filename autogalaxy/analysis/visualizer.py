import os
from os import path
from typing import List, Optional, Union

from autoconf import conf
import autoarray as aa
import autoarray.plot as aplt

from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.galaxy.plot.galaxy_plotters import GalaxyPlotter
from autogalaxy.galaxy.plot.hyper_galaxy_plotters import HyperPlotter
from autogalaxy.plane.plane import Plane
from autogalaxy.plane.plot.plane_plotters import PlanePlotter

from autogalaxy.plot.mat_wrap.include import Include2D
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot1D
from autogalaxy.plot.mat_wrap.mat_plot import MatPlot2D


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


class Visualizer:
    def __init__(self, visualize_path: str):
        """
        Visualizes a model-fit, including components of the model and fit objects.

        The `Visualizer` is typically used in the `Analysis` class of a non-linear search to visualize the maximum
        log likelihood model of the model-fit so far.

        The methods of the `Visualizer` are called throughout a non-linear search using the `Analysis`
        classes `visualize` method.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini`.

        Parameters
        ----------
        visualize_path
            The path on the hard-disk to the `image` folder of the non-linear searches results.
        """
        self.visualize_path = visualize_path

        self.plot_fit_no_hyper = plot_setting("hyper", "fit_no_hyper")

        self.include_2d = Include2D()

        try:
            os.makedirs(visualize_path)
        except FileExistsError:
            pass

    def mat_plot_1d_from(self, subfolders: str, format: str = "png") -> MatPlot1D:
        """
        Returns a 1D matplotlib plotting object whose `Output` class uses the `visualizer_path`, such that it outputs
        images to the `image` folder of the non-linear search.

        Parameters
        ----------
        subfolders
            Subfolders between the `image` folder of the non-linear search and where the images are output. For example,
            images associsted with a fit are output to the subfolder `fit`.
        format
            The format images are output as, e.g. `.png` files.

        Returns
        -------
        MatPlot1D
            The 1D matplotlib plotter object.
        """
        return MatPlot1D(
            output=aplt.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def mat_plot_2d_from(self, subfolders, format="png") -> MatPlot2D:
        """
        Returns a 2D matplotlib plotting object whose `Output` class uses the `visualizer_path`, such that it outputs
        images to the `image` folder of the non-linear search.

        Parameters
        ----------
        subfolders
            Subfolders between the `image` folder of the non-linear search and where the images are output. For example,
            images associsted with a fit are output to the subfolder `fit`.
        format
            The format images are output as, e.g. `.png` files.

        Returns
        -------
        MatPlot2D
            The 2D matplotlib plotter object.
        """
        return MatPlot2D(
            output=aplt.Output(
                path=path.join(self.visualize_path, subfolders), format=format
            )
        )

    def visualize_imaging(self, imaging: aa.Imaging):
        """
        Visualizes an `Imaging` dataset object.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `imaging`. When used with
        a non-linear search the `visualize_path` points to the search's results folder.
        `.
        Visualization includes individual images of attributes of the dataset (e.g. the image, noise map, PSF) and a
        subplot of all these attributes on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [dataset] header.

        Parameters
        ----------
        imaging
            The imaging dataset whose attributes are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["dataset", "imaging"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="imaging")

        imaging_plotter = aplt.ImagingPlotter(
            imaging=imaging, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        imaging_plotter.figures_2d(
            image=should_plot("data"),
            noise_map=should_plot("noise_map"),
            psf=should_plot("psf"),
            inverse_noise_map=should_plot("inverse_noise_map"),
            signal_to_noise_map=should_plot("signal_to_noise_map"),
            absolute_signal_to_noise_map=should_plot("absolute_signal_to_noise_map"),
            potential_chi_squared_map=should_plot("potential_chi_squared_map"),
        )

        if should_plot("subplot_dataset"):

            imaging_plotter.subplot_imaging()

    def visualize_interferometer(self, interferometer: aa.Interferometer):
        """
        Visualizes an `Interferometer` dataset object.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `interferometer`. When
        used with a non-linear search the `visualize_path` points to the search's results folder.

        Visualization includes individual images of attributes of the dataset (e.g. the visibilities, noise map,
        uv-wavelengths) and a subplot of all these attributes on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [dataset] header.

        Parameters
        ----------
        interferometer
            The interferometer dataset whose attributes are visualized.
        """

        def should_plot(name):
            return plot_setting(section=["dataset", "interferometer"], name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="interferometer")

        interferometer_plotter = aplt.InterferometerPlotter(
            interferometer=interferometer,
            include_2d=self.include_2d,
            mat_plot_2d=mat_plot_2d,
        )

        if should_plot("subplot_dataset"):
            interferometer_plotter.subplot_interferometer()

        interferometer_plotter.figures_2d(
            visibilities=should_plot("data"),
            u_wavelengths=should_plot("uv_wavelengths"),
            v_wavelengths=should_plot("uv_wavelengths"),
            amplitudes_vs_uv_distances=should_plot("amplitudes_vs_uv_distances"),
            phases_vs_uv_distances=should_plot("phases_vs_uv_distances"),
            dirty_image=should_plot("dirty_image"),
            dirty_noise_map=should_plot("dirty_noise_map"),
            dirty_signal_to_noise_map=should_plot("dirty_signal_to_noise_map"),
            dirty_inverse_noise_map=should_plot("dirty_inverse_noise_map"),
        )

    def visualize_plane(
        self, plane: Plane, grid: aa.type.Grid2DLike, during_analysis: bool
    ):
        """
        Visualizes a `Plane` object.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `plane`. When
        used with a non-linear search the `visualize_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `Plane` inferred by the search so far.

        Visualization includes individual images of attributes of the plane (e.g. its image, convergence, deflection
        angles) and a subplot of all these attributes on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [plane] header.

        Parameters
        ----------
        plane
            The maximum log likelihood `Plane` of the non-linear search.
        grid
            A 2D grid of (y,x) arc-second coordinates used to perform ray-tracing, which is the masked grid tied to
            the dataset.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        """

        def should_plot(name):
            return plot_setting(section="plane", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="plane")

        plane_plotter = PlanePlotter(
            plane=plane, grid=grid, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("subplot_plane"):

            plane_plotter.subplot()

        if should_plot("subplot_galaxy_images"):

            plane_plotter.subplot_galaxy_images()

        plane_plotter.figures_2d(
            image=should_plot("image"),
            convergence=should_plot("convergence"),
            potential=should_plot("potential"),
            deflections_y=should_plot("deflections"),
            deflections_x=should_plot("deflections"),
            magnification=should_plot("magnification"),
        )

        if not during_analysis:

            if should_plot("all_at_end_png"):

                plane_plotter.figures_2d(
                    image=True,
                    convergence=True,
                    potential=True,
                    deflections_y=True,
                    deflections_x=True,
                    magnification=True,
                )

            if should_plot("all_at_end_fits"):

                fits_mat_plot_2d = self.mat_plot_2d_from(
                    subfolders=path.join("plane", "fits"), format="fits"
                )

                plane_plotter = PlanePlotter(
                    plane=plane,
                    grid=grid,
                    mat_plot_2d=fits_mat_plot_2d,
                    include_2d=self.include_2d,
                )

                plane_plotter.figures_2d(
                    image=True,
                    convergence=True,
                    potential=True,
                    deflections_y=True,
                    deflections_x=True,
                    magnification=True,
                )

    def visualize_galaxies(
        self, galaxies: [List[Galaxy]], grid: aa.type.Grid2DLike, during_analysis: bool
    ):
        """
        Visualizes a list of `Galaxy` objects.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `galaxies`. When
        used with a non-linear search the `visualize_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `Galaxy`'s inferred by the search so far.

        Visualization includes individual images of attributes of each galaxy (e.g. 1D plots of their image,
        convergence) and a subplot of all these attributes on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [galaxies] header.

        Parameters
        ----------
        galaxies
            A list of the maximum log likelihood `Galaxy`'s of the non-linear search.
        grid
            A 2D grid of (y,x) arc-second coordinates used to perform ray-tracing, which is the masked grid tied to
            the dataset.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        """

        def should_plot(name):
            return plot_setting(section="galaxies", name=name)

        mat_plot_1d = self.mat_plot_1d_from(subfolders="galaxies")

        for galaxy in galaxies:

            galaxy_plotter = GalaxyPlotter(
                galaxy=galaxy,
                grid=grid,
                mat_plot_1d=mat_plot_1d,
                include_2d=self.include_2d,
            )

            try:
                galaxy_plotter.figures_1d_decomposed(
                    image=should_plot("image"),
                    convergence=should_plot("convergence"),
                    potential=should_plot("potential"),
                )
            except OverflowError:
                pass

    def visualize_inversion(self, inversion: aa.Inversion, during_analysis: bool):
        """
        Visualizes an `Inversion` object.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `inversion`. When
        used with a non-linear search the `visualize_path` points to the search's results folder and this function
        visualizes the maximum log likelihood `Inversion` inferred by the search so far.

        Visualization includes individual images of attributes of the dataset (e.g. the reconstructed image, the
        reconstruction) and a subplot of all these attributes on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [inversion] header.

        Parameters
        ----------
        inversion
            The inversion used to fit the dataset whose attributes are visualized.
        during_analysis
            Whether visualization is performed during a non-linear search or once it is completed.
        """

        def should_plot(name):
            return plot_setting(section="inversion", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="inversion")

        inversion_plotter = aplt.InversionPlotter(
            inversion=inversion, mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        inversion_plotter.figures_2d(
            reconstructed_image=should_plot("reconstructed_image")
        )

        inversion_plotter.figures_2d_of_mapper(
            mapper_index=0,
            reconstructed_image=should_plot("reconstructed_image"),
            reconstruction=should_plot("reconstruction"),
            errors=should_plot("errors"),
            residual_map=should_plot("residual_map"),
            normalized_residual_map=should_plot("normalized_residual_map"),
            chi_squared_map=should_plot("chi_squared_map"),
            regularization_weights=should_plot("regularization_weights"),
        )

        if should_plot("subplot_inversion"):
            for mapper_index in range(len(inversion.linear_obj_list)):
                inversion_plotter.subplot_of_mapper(mapper_index=mapper_index)

        if not during_analysis:

            if should_plot("all_at_end_png"):

                inversion_plotter.figures_2d(reconstructed_image=True)

                inversion_plotter.figures_2d_of_mapper(
                    mapper_index=0,
                    reconstructed_image=True,
                    reconstruction=True,
                    errors=True,
                    residual_map=True,
                    normalized_residual_map=True,
                    chi_squared_map=True,
                    regularization_weights=True,
                )

    def visualize_hyper_images(
        self,
        hyper_galaxy_image_path_dict: {str, aa.Array2D},
        hyper_model_image: aa.Array2D,
    ):
        """
        Visualizes the hyper-images and hyper dataset inferred by a model-fit.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `hyper`. When
        used with a non-linear search the `visualize_path` points to the search's results folder.

        Visualization includes individual images of attributes of the hyper dataset (e.g. the hyper model image) and
        a subplot of all hyper galaxy images on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [hyper] header.

        Parameters
        ----------
        hyper_galaxy_image_path_dict
            A dictionary mapping the path to each galaxy (e.g. its name) to its corresponding hyper galaxy image.
        hyper_model_image
            The hyper model image which corresponds to the sum of hyper galaxy images.
        """

        def should_plot(name):
            return plot_setting(section="hyper", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="hyper")

        hyper_plotter = HyperPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if should_plot("model_image"):
            hyper_plotter.figure_hyper_model_image(hyper_model_image=hyper_model_image)

        if should_plot("images_of_galaxies"):

            hyper_plotter.subplot_hyper_images_of_galaxies(
                hyper_galaxy_image_path_dict=hyper_galaxy_image_path_dict
            )

    def visualize_contribution_maps(self, plane: Plane):
        """
        Visualizes the contribution maps that are used for hyper features which adapt a model to the dataset it is
        fitting.

        Images are output to the `image` folder of the `visualize_path` in a subfolder called `hyper`. When
        used with a non-linear search the `visualize_path` points to the search's results folder and this function
        visualizes the maximum log likelihood contribution maps inferred by the search so far.

        Visualization includes individual images of attributes of the hyper dataset (e.g. the contribution map of
        each galaxy) and a subplot of all contribution maps on the same figure.

        The images output by the `Visualizer` are customized using the file `config/visualize/plots.ini` under the
        [hyper] header.

        Parameters
        ----------
        plane
            The maximum log likelihood `Plane` of the non-linear search which is used to plot the contribution maps.
        """

        def should_plot(name):
            return plot_setting(section="hyper", name=name)

        mat_plot_2d = self.mat_plot_2d_from(subfolders="hyper")

        hyper_plotter = HyperPlotter(
            mat_plot_2d=mat_plot_2d, include_2d=self.include_2d
        )

        if hasattr(plane, "contribution_map_list"):
            if should_plot("contribution_map_list"):
                hyper_plotter.subplot_contribution_map_list(
                    contribution_map_list_list=plane.contribution_map_list
                )
