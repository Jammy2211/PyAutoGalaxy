import copy
import numpy as np
from typing import Dict, List, Tuple, Type, Union

from autoconf import cached_property
from autoconf import conf
import autofit as af
import autoarray as aa

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

        instance = self.analysis.instance_with_associated_adapt_images_from(
            instance=self.instance
        )

        return self.analysis.plane_via_instance_from(instance=instance)

    @property
    def path_galaxy_tuples(self) -> List[Tuple[str, Galaxy]]:
        """
        Tuples associating the names of galaxies with instances from the best fit
        """
        return self.instance.path_instance_tuples_for_class(cls=Galaxy)


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
    def adapt_galaxy_image_path_dict(self) -> Dict[str, aa.Array2D]:
        """
        A dictionary associating 1D galaxy images with their names.
        """

        adapt_minimum_percent = conf.instance["general"]["adapt"][
            "adapt_minimum_percent"
        ]

        adapt_galaxy_image_path_dict = {}

        for path, galaxy in self.path_galaxy_tuples:
            galaxy_image = self.image_galaxy_dict[path]

            if not np.all(galaxy_image == 0):
                minimum_galaxy_value = adapt_minimum_percent * max(galaxy_image)
                galaxy_image[galaxy_image < minimum_galaxy_value] = minimum_galaxy_value

            adapt_galaxy_image_path_dict[path] = galaxy_image

        return adapt_galaxy_image_path_dict

    @property
    def adapt_model_image(self) -> aa.Array2D:
        """
        The adapt image used by Analysis objects to adapt aspects of a model to the dataset being fitted.

        The adapt image is the sum of the galaxy image of every individual galaxy.
        """
        adapt_model_image = aa.Array2D(
            values=np.zeros(self.mask.derive_mask.sub_1.pixels_in_mask),
            mask=self.mask.derive_mask.sub_1,
        )

        for path, galaxy in self.path_galaxy_tuples:
            adapt_model_image += self.adapt_galaxy_image_path_dict[path]

        return adapt_model_image
