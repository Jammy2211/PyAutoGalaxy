from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type, Union

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autogalaxy.analysis.analysis import AnalysisDataset

from autoconf import cached_property

import autofit as af
import autoarray as aa

from autogalaxy.analysis.adapt_images import AdaptImages
from autogalaxy.galaxy.galaxy import Galaxy


class Result(af.Result):
    def __init__(
            self,
            samples_summary : af.SamplesSummary,
            paths : af.AbstractPaths,
            samples: Optional[af.SamplesPDF] = None,
            search_internal : Optional[object] = None,
            analysis : Optional[AnalysisDataset] = None
    ):
        """
        After the non-linear search of a fit to a dataset is complete it creates a `Result` object which includes:

        - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
          the maximum likelihood model, posteriors and other properties.

        - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
          an instance of the maximum log likelihood model).

        - The non-linear search used to perform the model fit.

        This class contains a number of methods which use the above objects to create the max log likelihood galaxies,
        `FitIamging`, adapt-galaxy images,etc.

        Parameters
        ----------
        samples
            A PyAutoFit object which contains the samples of the non-linear search, for example the chains of an MCMC
            run of samples of the nested sampler.
        model
            The PyAutoFit model object, which includes model components representing the galaxies that are fitted to
            the imaging data.
        analysis
            The `Analysis` object that was used to perform the model-fit from which this result is inferred.
        search
            The non-linear search used to perform this model-fit.

        Returns
        -------
        ResultImaging
            The result of fitting the model to the imaging dataset, via a non-linear search.
        """
        super().__init__(samples_summary=samples_summary, paths=paths, samples=samples, search_internal=search_internal)

        self.analysis = analysis

    @property
    def max_log_likelihood_fit(self):
        raise NotImplementedError

    @property
    def max_log_likelihood_galaxies(self) -> List[Galaxy]:
        """
        An instance of the list of galaxies corresponding to the maximum log likelihood model inferred by the
        non-linear search.
        """
        return self.analysis.galaxies_via_instance_from(instance=self.instance)

    @property
    def path_galaxy_tuples(self) -> List[Tuple[str, Galaxy]]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        path_galaxy_tuples = []

        for path, galaxy in self.instance.path_instance_tuples_for_class(cls=Galaxy):
            path_galaxy_tuples.append((str(path), galaxy))

        return path_galaxy_tuples


class ResultDataset(Result):
    def cls_list_from(self, cls: Type) -> List:
        """
        A list of all pixelization classes used by the model-fit.
        """
        return self.max_log_likelihood_galaxies.cls_list_from(cls=cls)

    @property
    def mask(self) -> aa.Mask2D:
        """
        The 2D mask applied to the dataset for the model-fit.
        """
        return self.analysis.dataset.mask

    @property
    def grid(self) -> aa.Grid2D:
        """
        The masked 2D grid used by the dataset in the model-fit.
        """
        return self.analysis.dataset.grid

    @property
    def dataset(self) -> Union[aa.Imaging, aa.Interferometer]:
        """
        The dataset that was fitted by the model-fit.
        """
        return self.max_log_likelihood_fit.dataset

    @cached_property
    def model_image_galaxy_dict(self) -> Dict[str, Galaxy]:
        """
        A dictionary associating galaxy names with model images of those galaxies.

        This is used for creating the adapt-dataset used by Analysis objects to adapt aspects of a model to the dataset
        being fitted.
        """

        galaxy_model_image_dict = self.max_log_likelihood_fit.galaxy_model_image_dict

        return {
            galaxy_path: galaxy_model_image_dict[galaxy]
            for galaxy_path, galaxy in self.path_galaxy_tuples
        }

    @cached_property
    def subtracted_image_galaxy_dict(self) -> Dict[str, Galaxy]:
        """
        A dictionary associating galaxy names with subtracted images (the data minus all other galaxy images) of
        those galaxies.

        This is used for creating the adapt-dataset used by Analysis objects to adapt aspects of a subtracted to the
        dataset being fitted.
        """

        galaxy_subtracted_image_dict = (
            self.max_log_likelihood_fit.subtracted_images_of_galaxies_dict
        )

        return {
            galaxy_path: galaxy_subtracted_image_dict[galaxy]
            for galaxy_path, galaxy in self.path_galaxy_tuples
        }

    @cached_property
    def subtracted_signal_to_noise_map_galaxy_dict(self) -> Dict[str, Galaxy]:
        """
        A dictionary associating galaxy names with subtracted images (the data minus all other galaxy images) of
        those galaxies.

        This is used for creating the adapt-dataset used by Analysis objects to adapt aspects of a subtracted to the
        dataset being fitted.
        """

        galaxy_subtracted_signal_to_noise_map_dict = (
            self.max_log_likelihood_fit.subtracted_signal_to_noise_maps_of_galaxies_dict
        )

        return {
            galaxy_path: galaxy_subtracted_signal_to_noise_map_dict[galaxy]
            for galaxy_path, galaxy in self.path_galaxy_tuples
        }

    def adapt_images_from(self, use_model_images: bool = False) -> AdaptImages:
        """
        Returns the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
        reconstructed galaxy's morphology.

        This can use either:

        - The model image of each galaxy in the best-fit model.
        - The subtracted image of each galaxy in the best-fit model, where the subtracted image is the dataset
          minus the model images of all other galaxies.

        In **PyAutoLens** these adapt images have had lensing calculations performed on them and therefore for source
        galaxies are their lensed model images in the image-plane.

        Parameters
        ----------
        use_model_images
            If True, the model images of the galaxies are used to create the adapt images. If False, the subtracted
            images of the galaxies are used.
        """

        return AdaptImages.from_result(
            result=self,
            use_model_images=use_model_images,
        )
