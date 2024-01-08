import logging
from typing import Callable, Union

import autoarray as aa
import autofit as af
from autoconf import conf
from autofit.exc import PriorLimitException
from autogalaxy.analysis.preloads import Preloads

logger = logging.getLogger(__name__)

logger.setLevel(level="INFO")


class FitMaker:
    def __init__(self, model: af.Collection, fit_from: Callable):
        """
        Makes fits using an input PyAutoFit `model`, where the parameters of the model are drawn from its prior. This
        uses an input `fit_from`, which given an `instance` of the model creates the fit object.

        This is used for implicit preloading in the `Analysis` classes, whereby the created fits are compared against
        one another to determine whether certain components of the analysis can be preloaded.

        This includes functionality for creating the fit via the model in different ways, so that if certain
        models are ill-defined another is used instead.

        Parameters
        ----------
        model
            A **PyAutoFit** model object which via its parameters and their priors can created instances of the model.
        fit_from
            A function which given the instance of the model creates a `Fit` object.
        """

        self.model = model
        self.fit_from = fit_from

    @property
    def preloads_cls(self):
        return Preloads

    def fit_via_model_from(
        self, unit_value: float
    ) -> Union[aa.FitImaging, aa.FitInterferometer]:
        """
        Create a fit via the model.

        This first tries to compute the fit from the input `unit_value`, where the `unit_value` defines unit hyper
        cube values of each parameter's prior in the model, used to map each value to physical values for the fit.

        If this model fit produces an `Exception` because the parameter combination does not fit the data accurately,
        a sequence of random fits are instead used into an exception is not returned. However, if the number
        of `preload_attempts` defined in the configuration files is exceeded a None is returned.

        Parameters
        ----------
        unit_value
            The unit hyper cube values of each parameter's prior in the model, used to map each value to physical
            values for the fit.

        Returns
        -------
        fit
            A fit object where an instance of the model has been fitted to the data.
        """
        try:
            try:
                return self.fit_unit_instance_from(unit_value=unit_value)
            except IndexError as e:
                raise Exception from e
        except (Exception, PriorLimitException):
            return self.fit_random_instance_from()

    def fit_unit_instance_from(
        self, unit_value: float
    ) -> Union[aa.FitImaging, aa.FitInterferometer]:
        """
        Create a fit via the model using an input `unit_value`, where the `unit_value` defines unit hyper
        cube values of each parameter's prior in the model, used to map each value to physical values for the fit.

        Parameters
        ----------
        unit_value
            The unit hyper cube values of each parameter's prior in the model, used to map each value to physical
            values for the fit.

        Returns
        -------
        fit
            A fit object where an instance of the model has been fitted to the data.
        """
        instance = self.model.instance_from_unit_vector(
            unit_vector=[unit_value] * self.model.prior_count, ignore_prior_limits=True
        )

        fit = self.fit_from(
            instance=instance,
        )
        fit.figure_of_merit
        return fit

    def fit_random_instance_from(self) -> Union[aa.FitImaging, aa.FitInterferometer]:
        """
        Create a fit via the model by guessing a sequence of random fits until an exception is not returned. If
        the number of `preload_attempts` defined in the configuration files is exceeded a None is returned.

        Returns
        -------
        fit
            A fit object where an instance of the model has been fitted to the data.
        """

        preload_attempts = conf.instance["general"]["analysis"]["preload_attempts"]

        for i in range(preload_attempts):
            try:
                instance = self.model.random_instance(ignore_prior_limits=True)

                fit = self.fit_from(
                    instance=instance,
                )

                fit.figure_of_merit

                return fit

            except Exception as e:
                continue
