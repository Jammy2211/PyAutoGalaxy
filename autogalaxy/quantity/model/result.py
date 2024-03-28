from autogalaxy.analysis.result import Result
from autogalaxy.quantity.fit_quantity import FitQuantity


class ResultQuantity(Result):
    @property
    def max_log_likelihood_fit(self) -> FitQuantity:
        """
        An instance of a `FitQuantity` corresponding to the maximum log likelihood model inferred by the non-linear
        search.
        """

        return self.analysis.fit_quantity_for_instance(instance=self.instance)
