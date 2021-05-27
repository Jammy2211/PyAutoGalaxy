from astropy import cosmology as cosmo

import autofit as af
from autoarray import preloads as pload
from autoarray.exc import PixelizationException, InversionException, GridException
from autoarray.inversion import pixelizations as pix, inversions as inv
from autofit.exc import FitException
from autogalaxy.analysis import result as res
from autogalaxy.analysis import visualizer as vis
from autogalaxy.fit import fit_imaging, fit_interferometer
from autogalaxy.galaxy import galaxy as g
from autogalaxy.plane import plane as pl


class Analysis(af.Analysis):
    def __init__(self, hyper_result=None, cosmology=cosmo.Planck15):
        self.hyper_result = hyper_result
        self.cosmology = cosmology


class AnalysisDataset(Analysis):
    def __init__(
        self,
        dataset,
        hyper_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        preloads=pload.Preloads(),
    ):

        super().__init__(hyper_result=hyper_result, cosmology=cosmology)

        self.dataset = dataset

        if self.hyper_result is not None:

            if hyper_result.search is not None:
                hyper_result.search.paths = None

            self.set_hyper_dataset(result=self.hyper_result)

        else:

            self.hyper_galaxy_image_path_dict = None
            self.hyper_model_image = None

        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion
        self.preloads = preloads

    def set_hyper_dataset(self, result):

        self.hyper_galaxy_image_path_dict = result.hyper_galaxy_image_path_dict
        self.hyper_model_image = result.hyper_model_image

    def hyper_image_sky_for_instance(self, instance):

        if hasattr(instance, "hyper_image_sky"):
            return instance.hyper_image_sky

    def hyper_background_noise_for_instance(self, instance):

        if hasattr(instance, "hyper_background_noise"):
            return instance.hyper_background_noise

    def plane_for_instance(self, instance):
        return pl.Plane(galaxies=instance.galaxies)

    def associate_hyper_images(self, instance: af.ModelInstance) -> af.ModelInstance:
        """
        Takes images from the last result, if there is one, and associates them with galaxies in this search
        where full-path galaxy names match.

        If the galaxy collection has a different name then an association is not made.

        e.g.
        galaxies.lens will match with:
            galaxies.lens
        but not with:
            galaxies.lens
            galaxies.source

        Parameters
        ----------
        instance
            A model instance with 0 or more galaxies in its tree

        Returns
        -------
        instance
           The input instance with images associated with galaxies where possible.
        """

        if self.hyper_galaxy_image_path_dict is not None:

            for galaxy_path, galaxy in instance.path_instance_tuples_for_class(
                g.Galaxy
            ):
                if galaxy_path in self.hyper_galaxy_image_path_dict:
                    galaxy.hyper_model_image = self.hyper_model_image

                    galaxy.hyper_galaxy_image = self.hyper_galaxy_image_path_dict[
                        galaxy_path
                    ]

        return instance

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        paths.save_object("data", self.dataset.data)
        paths.save_object("noise_map", self.dataset.noise_map)
        paths.save_object("settings_dataset", self.dataset.settings)
        paths.save_object("settings_inversion", self.settings_inversion)
        paths.save_object("settings_pixelization", self.settings_pixelization)

        paths.save_object("cosmology", self.cosmology)

        if self.hyper_model_image is not None:
            paths.save_object("hyper_model_image", self.hyper_model_image)

        if self.hyper_galaxy_image_path_dict is not None:
            paths.save_object(
                "hyper_galaxy_image_path_dict", self.hyper_galaxy_image_path_dict
            )


class AnalysisImaging(AnalysisDataset):
    def __init__(
        self,
        dataset,
        hyper_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        preloads=pload.Preloads(),
    ):

        super().__init__(
            dataset=dataset,
            hyper_result=hyper_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
        )

        self.dataset = dataset

    @property
    def imaging(self):
        return self.dataset

    def log_likelihood_function(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the imaging in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model imaging itself
        """

        self.associate_hyper_images(instance=instance)
        plane = self.plane_for_instance(instance=instance)

        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        try:
            fit = self.fit_imaging_for_plane(
                plane=plane,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            return fit.figure_of_merit
        except (PixelizationException, InversionException, GridException) as e:
            raise FitException from e

    def fit_imaging_for_plane(
        self, plane, hyper_image_sky, hyper_background_noise, use_hyper_scalings=True
    ):

        return fit_imaging.FitImaging(
            imaging=self.dataset,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scalings=use_hyper_scalings,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
        )

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis):

        instance = self.associate_hyper_images(instance=instance)
        plane = self.plane_for_instance(instance=instance)
        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.fit_imaging_for_plane(
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        visualizer = vis.Visualizer(visualize_path=paths.image_path)
        visualizer.visualize_imaging(imaging=self.imaging)
        visualizer.visualize_fit_imaging(fit=fit, during_analysis=during_analysis)

        if fit.inversion is not None:
            visualizer.visualize_inversion(
                inversion=fit.inversion, during_analysis=during_analysis
            )

        visualizer.visualize_hyper_images(
            hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
            hyper_model_image=self.hyper_model_image,
            plane=plane,
        )

        if visualizer.plot_fit_no_hyper:

            fit = self.fit_imaging_for_plane(
                plane=plane,
                hyper_image_sky=None,
                hyper_background_noise=None,
                use_hyper_scalings=False,
            )

            visualizer.visualize_fit_imaging(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return res.ResultImaging(
            samples=samples, model=model, analysis=self, search=search
        )

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        super().save_attributes_for_aggregator(paths=paths)

        paths.save_object("psf", self.dataset.psf_unormalized)
        paths.save_object("mask", self.dataset.mask)


class AnalysisInterferometer(AnalysisDataset):
    def __init__(
        self,
        dataset,
        hyper_result=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        preloads=pload.Preloads(),
    ):

        super().__init__(
            dataset=dataset,
            hyper_result=hyper_result,
            cosmology=cosmology,
            settings_pixelization=settings_pixelization,
            settings_inversion=settings_inversion,
            preloads=preloads,
        )

        if self.hyper_result is not None:

            self.set_hyper_dataset(result=self.hyper_result)

        else:

            self.hyper_galaxy_visibilities_path_dict = None
            self.hyper_model_visibilities = None

    def set_hyper_dataset(self, result):

        super().set_hyper_dataset(result=result)

        self.hyper_model_visibilities = result.hyper_model_visibilities
        self.hyper_galaxy_visibilities_path_dict = (
            result.hyper_galaxy_visibilities_path_dict
        )

    @property
    def interferometer(self):
        return self.dataset

    def log_likelihood_function(self, instance):
        """
        Determine the fit of a lens galaxy and source galaxy to the interferometer in this lens.

        Parameters
        ----------
        instance
            A model instance with attributes

        Returns
        -------
        fit : Fit
            A fractional value indicating how well this model fit and the model interferometer itself
        """

        self.associate_hyper_images(instance=instance)
        plane = self.plane_for_instance(instance=instance)

        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        try:
            fit = self.fit_interferometer_for_plane(
                plane=plane, hyper_background_noise=hyper_background_noise
            )

            return fit.figure_of_merit
        except (PixelizationException, InversionException, GridException) as e:
            raise FitException from e

    def associate_hyper_visibilities(
        self, instance: af.ModelInstance
    ) -> af.ModelInstance:
        """
        Takes visibilities from the last result, if there is one, and associates them with galaxies in this search
        where full-path galaxy names match.

        If the galaxy collection has a different name then an association is not made.

        e.g.
        galaxies.lens will match with:
            galaxies.lens
        but not with:
            galaxies.lens
            galaxies.source

        Parameters
        ----------
        instance
            A model instance with 0 or more galaxies in its tree

        Returns
        -------
        instance
           The input instance with visibilities associated with galaxies where possible.
        """
        if self.hyper_galaxy_visibilities_path_dict is not None:
            for galaxy_path, galaxy in instance.path_instance_tuples_for_class(
                g.Galaxy
            ):
                if galaxy_path in self.hyper_galaxy_visibilities_path_dict:
                    galaxy.hyper_model_visibilities = self.hyper_model_visibilities
                    galaxy.hyper_galaxy_visibilities = self.hyper_galaxy_visibilities_path_dict[
                        galaxy_path
                    ]

        return instance

    def fit_interferometer_for_plane(
        self, plane, hyper_background_noise, use_hyper_scalings=True
    ):

        return fit_interferometer.FitInterferometer(
            interferometer=self.dataset,
            plane=plane,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scalings=use_hyper_scalings,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
        )

    def visualize(self, paths: af.DirectoryPaths, instance, during_analysis):

        self.associate_hyper_images(instance=instance)
        plane = self.plane_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.fit_interferometer_for_plane(
            plane=plane, hyper_background_noise=hyper_background_noise
        )

        visualizer = vis.Visualizer(visualize_path=paths.image_path)
        visualizer.visualize_interferometer(interferometer=self.interferometer)
        visualizer.visualize_fit_interferometer(
            fit=fit, during_analysis=during_analysis
        )
        if fit.inversion is not None:
            visualizer.visualize_inversion(
                inversion=fit.inversion, during_analysis=during_analysis
            )

        visualizer.visualize_hyper_images(
            hyper_galaxy_image_path_dict=self.hyper_galaxy_image_path_dict,
            hyper_model_image=self.hyper_model_image,
            plane=plane,
        )

        if visualizer.plot_fit_no_hyper:
            fit = self.fit_interferometer_for_plane(
                plane=plane, hyper_background_noise=None, use_hyper_scalings=False
            )

            visualizer.visualize_fit_interferometer(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def make_result(
        self, samples: af.PDFSamples, model: af.Collection, search: af.NonLinearSearch
    ):
        return res.ResultInterferometer(
            samples=samples, model=model, analysis=self, search=search
        )

    def save_attributes_for_aggregator(self, paths: af.DirectoryPaths):

        super().save_attributes_for_aggregator(paths=paths)

        paths.save_object("uv_wavelengths", self.dataset.uv_wavelengths)
        paths.save_object("real_space_mask", self.dataset.real_space_mask)
