from typing import List

import autofit as af
import autoarray as aa

from autogalaxy.ellipse.fit_ellipse import FitEllipse
from autogalaxy.ellipse.model.result import ResultEllipse
from autogalaxy.ellipse.model.visualizer import VisualizerEllipse

class AnalysisEllipse(af.Analysis):

    Result = ResultEllipse
    Visualizer = VisualizerEllipse

    def __init__(self, dataset: aa.Imaging):
        """
        Fits a model made of ellipses to an imaging dataset via a non-linear search.

        The `Analysis` class defines the `log_likelihood_function` which fits the model to the dataset and returns the
        log likelihood value defining how well the model fitted the data.

        It handles many other tasks, such as visualization, outputting results to hard-disk and storing results in
        a format that can be loaded after the model-fit is complete.

        This class is used for model-fits which fit ellipses to an imaging dataset.

        Parameters
        ----------
        dataset
            The `Imaging` dataset that the model containing ellipses is fitted to.
        """
        self.dataset = dataset

    def log_likelihood_function(self, instance: af.ModelInstance) -> float:
        """
        Given an instance of the model, where the model parameters are set via a non-linear search, fit the model
        instance to the imaging dataset.

        This function returns a log likelihood which is used by the non-linear search to guide the model-fit.

        For this analysis class, this function performs the following steps:

        1) Extract all ellipses from the model instance.

        2) Use the ellipses to create a list of `FitEllipse` objects, which fits each ellipse to the data and noise-map
        via interpolation and subtracts these values from their mean values in order to quantify how well the ellipse
        traces around the data.

        Certain models will fail to fit the dataset and raise an exception. For example the ellipse parameters may be
        ill defined and raise an Exception. In such circumstances the model is discarded and its likelihood value is
        passed to the non-linear search in a way that it ignores it (for example, using a value of -1.0e99).

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        float
            The log likelihood indicating how well this model instance fitted the imaging data.
        """
        fit_list = self.fit_list_from(instance=instance)

        return sum(fit.log_likelihood for fit in fit_list)

    def fit_list_from(self, instance: af.ModelInstance) -> List[FitEllipse]:
        """
        Given a model instance create a list of `FitEllipse` objects.

        This function is used in the `log_likelihood_function` to fit the model containing ellipses to the imaging data
        and compute the log likelihood.

        Parameters
        ----------
        instance
            An instance of the model that is being fitted to the data by this analysis (whose parameters have been set
            via a non-linear search).

        Returns
        -------
        The fit of the ellipses to the imaging dataset, which includes the log likelihood.
        """
        return [FitEllipse(dataset=self.dataset, ellipse=ellipse) for ellipse in instance.ellipses]

