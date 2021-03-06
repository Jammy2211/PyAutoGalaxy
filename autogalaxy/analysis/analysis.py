from astropy import cosmology as cosmo
import autofit as af
from autoarray.exc import PixelizationException, InversionException, GridException
from autofit.exc import FitException
from autoarray.inversion import pixelizations as pix, inversions as inv
from autogalaxy.galaxy import galaxy as g
from autogalaxy.plane import plane as pl
from autogalaxy.fit import fit
from autogalaxy.analysis import visualizer as vis
from autogalaxy.analysis import result as res


class Analysis(af.Analysis):
    def __init__(self, cosmology=cosmo.Planck15):

        self.cosmology = cosmology

    # def save_settings(self, paths: af.Paths):
    #     with open(f"{paths.pickle_path}/settings.pickle", "wb+") as f:
    #         pickle.dump(self.settings, f)

    def modify_before_fit(self, model, paths: af.Paths):
        return self


class AnalysisDataset(Analysis):
    def __init__(
        self,
        dataset,
        results=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        preloads=None,
    ):

        super().__init__(cosmology=cosmology)

        self.dataset = dataset

        result = res.last_result_with_use_as_hyper_dataset(results=results)

        if result is not None:

            self.hyper_galaxy_image_path_dict = result.hyper_galaxy_image_path_dict
            self.hyper_model_image = result.hyper_model_image

        else:

            self.hyper_galaxy_image_path_dict = None
            self.hyper_model_image = None

        self.settings_pixelization = settings_pixelization
        self.settings_inversion = settings_inversion
        self.preloads = preloads

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
        Takes images from the last result, if there is one, and associates them with galaxies in this phase
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


class AnalysisImaging(AnalysisDataset):
    def __init__(
        self,
        dataset,
        results=None,
        cosmology=cosmo.Planck15,
        settings_pixelization=pix.SettingsPixelization(),
        settings_inversion=inv.SettingsInversion(),
        preloads=None,
    ):

        super().__init__(
            dataset=dataset,
            results=results,
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
            fit = self.imaging_fit_for_plane(
                plane=plane,
                hyper_image_sky=hyper_image_sky,
                hyper_background_noise=hyper_background_noise,
            )

            return fit.figure_of_merit
        except (PixelizationException, InversionException, GridException) as e:
            raise FitException from e

    def imaging_fit_for_plane(
        self, plane, hyper_image_sky, hyper_background_noise, use_hyper_scalings=True
    ):

        return fit.FitImaging(
            masked_imaging=self.dataset,
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
            use_hyper_scalings=use_hyper_scalings,
            settings_pixelization=self.settings_pixelization,
            settings_inversion=self.settings_inversion,
        )

    def visualize(self, paths: af.Paths, instance, during_analysis):

        instance = self.associate_hyper_images(instance=instance)
        plane = self.plane_for_instance(instance=instance)
        hyper_image_sky = self.hyper_image_sky_for_instance(instance=instance)
        hyper_background_noise = self.hyper_background_noise_for_instance(
            instance=instance
        )

        fit = self.imaging_fit_for_plane(
            plane=plane,
            hyper_image_sky=hyper_image_sky,
            hyper_background_noise=hyper_background_noise,
        )

        visualizer = vis.Visualizer(visualize_path=paths.image_path)
        visualizer.visualize_imaging(imaging=self.imaging.imaging)
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

            fit = self.imaging_fit_for_plane(
                plane=plane,
                hyper_image_sky=None,
                hyper_background_noise=None,
                use_hyper_scalings=False,
            )

            visualizer.visualize_fit_imaging(
                fit=fit, during_analysis=during_analysis, subfolders="fit_no_hyper"
            )

    def make_result(
        self,
        samples: af.PDFSamples,
        model: af.CollectionPriorModel,
        search: af.NonLinearSearch,
    ):
        return res.ResultDataset(
            samples=samples, model=model, analysis=self, search=search
        )
