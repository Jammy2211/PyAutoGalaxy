from typing import Dict, List, Tuple, Type, Union

from autoconf import cached_property

import autofit as af
import autoarray as aa

from autogalaxy.analysis.adapt_images import AdaptImages
from autogalaxy.galaxy.galaxy import Galaxy
from autogalaxy.plane.plane import Plane


class Result(af.Result):
    def __init__(self, samples: af.SamplesPDF, analysis):
        """
        After the non-linear search of a fit to a dataset is complete it creates a `Result` object which includes:

        - The samples of the non-linear search (E.g. MCMC chains, nested sampling samples) which are used to compute
          the maximum likelihood model, posteriors and other properties.

        - The model used to fit the data, which uses the samples to create specific instances of the model (e.g.
          an instance of the maximum log likelihood model).

        - The non-linear search used to perform the model fit.

        This class contains a number of methods which use the above objects to create the max log likelihood `Plane`,
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
        super().__init__(samples=samples)

        self.analysis = analysis

    @property
    def max_log_likelihood_fit(self):
        raise NotImplementedError

    @property
    def max_log_likelihood_plane(self) -> Plane:
        """
        An instance of a `Plane` corresponding to the maximum log likelihood model inferred by the non-linear search.
        """
        return self.analysis.plane_via_instance_from(instance=self.instance)

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
        return self.max_log_likelihood_plane.cls_list_from(cls=cls)

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
    def image_galaxy_dict(self) -> Dict[str, Galaxy]:
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

    @property
    def adapt_images(self) -> AdaptImages:
        """
        Returns the adapt-images which are used to make a pixelization's mesh and regularization adapt to the
        reconstructed galaxy's morphology.
        """

        return AdaptImages.from_result(
            result=self,
        )
