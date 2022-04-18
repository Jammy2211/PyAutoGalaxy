from abc import ABC, abstractmethod
from functools import partial
from typing import List, Generator

import autofit as af


class AbstractAgg(ABC):
    def __init__(self, aggregator: af.Aggregator):
        """
        An abstract aggregator wrapper, which makes it straight forward to compute generators of objects from specific
        samples of a non-linear search.

        For example, in **PyAutoLens**, this makes it straight forward to create generators of `Plane`'s drawn from
        the PDF estimated by a non-linear for efficient error calculation of derived quantities.
        Parameters
        ----------
        aggregator
            An PyAutoFit aggregator containing the results of non-linear searches performed by PyAutoFit.
        """
        self.aggregator = aggregator

    @abstractmethod
    def make_object_for_gen(self, fit: af.Fit, galaxies: List["Galaxy"]) -> object:
        """
        For example, in the `PlaneAgg` object, this function is overwritten such that it creates a `Plane` from a
        `ModelInstance` that contains the galaxies of a sample from a non-linear search.

        Parameters
        ----------
        fit
            A PyAutoFit database Fit object containing the generators of the results of PyAutoGalaxy model-fits.
        galaxies
            A list of galaxies corresponding to a sample of a non-linear search and model-fit.

        Returns
        -------
        Generator
            A generator that creates an object used in the model-fitting process of a non-linear search.
        """

    def max_log_likelihood_gen_from(self) -> Generator:
        """
        Returns a generator using the maximum likelihood instance of a non-linear search.

        This generator creates a list containing the maximum log instance of every result loaded in the aggregator.

        For example, in **PyAutoLens**, by overwriting the `make_gen_from` method this returns a generator
        of `Plane` objects from a PyAutoFit aggregator. This generator then generates a list of the maximum log
        likelihood `Plane` objects for all aggregator results.
        """

        def func_gen(fit: af.Fit) -> Generator:
            return self.make_object_for_gen(fit=fit, galaxies=fit.instance.galaxies)

        return self.aggregator.map(func=func_gen)

    def weights_above_gen_from(self, minimum_weight: float) -> List:
        """
        Returns a list of all weights above a minimum weight for every result.

        Parameters
        ----------
        minimum_weight
            The minimum weight of a non-linear sample, such that samples with a weight below this value are discarded
            and not included in the generator.
        """

        def func_gen(fit: af.Fit, minimum_weight: float) -> List[object]:

            samples = fit.value(name="samples")

            weight_list = []

            for sample in samples.sample_list:

                if sample.weight > minimum_weight:

                    weight_list.append(sample.weight)

            return weight_list

        func = partial(func_gen, minimum_weight=minimum_weight)

        return self.aggregator.map(func=func)

    def all_above_weight_gen_from(self, minimum_weight: float) -> Generator:
        """
        Returns a generator which for every result generates a list of objects whose parameter values are all those
        in the non-linear search with a weight about an input `minimum_weight` value. This enables straight forward
        error estimation.

        This generator creates lists containing instances whose non-linear sample weight are above the value of
        `minimum_weight`. For example, if the aggregator contains 10 results and each result has 100 samples above the
        `minimum_weight`, a list of 10 entries will be returned, where each entry in this list contains 100 object's
        paired with each non-linear sample.

        For example, in **PyAutoLens**, by overwriting the `make_gen_from` method this returns a generator
        of `Plane` objects from a PyAutoFit aggregator. This generator then generates lists of `Plane` objects
        corresponding to all non-linear search samples above the `minimum_weight`.

        Parameters
        ----------
        minimum_weight
            The minimum weight of a non-linear sample, such that samples with a weight below this value are discarded
            and not included in the generator.
        """

        def func_gen(fit: af.Fit, minimum_weight: float) -> List[object]:

            samples = fit.value(name="samples")

            all_above_weight_list = []

            for sample in samples.sample_list:

                if sample.weight > minimum_weight:
                    instance = sample.instance_for_model(model=samples.model)

                    all_above_weight_list.append(
                        self.make_object_for_gen(fit=fit, galaxies=instance.galaxies)
                    )

            return all_above_weight_list

        func = partial(func_gen, minimum_weight=minimum_weight)

        return self.aggregator.map(func=func)

    def randomly_drawn_via_pdf_gen_from(self, total_samples: int):
        """
        Returns a generator which for every result generates a list of objects whose parameter values are drawn
        randomly from the PDF. This enables straight forward error estimation.

        This generator creates lists containing instances that are drawn randomly from the PDF for every result loaded
        in the aggregator. For example, the aggregator contains 10 results and if `total_samples=100`, a list of 10
        entries will be returned, where each entry in this list contains 100 object's paired with non-linear samples
        randomly drawn from the PDF.

        For example, in **PyAutoLens**, by overwriting the `make_gen_from` method this returns a generator
        of `Plane` objects from a PyAutoFit aggregator. This generator then generates lists of `Plane` objects
        corresponding to non-linear search samples randomly drawn from the PDF.

        Parameters
        ----------
        total_samples
            The total number of non-linear search samples that should be randomly drawn from the PDF.
        """

        def func_gen(fit: af.Fit, total_samples: int) -> List[object]:

            samples = fit.value(name="samples")

            return [
                self.make_object_for_gen(
                    fit=fit,
                    galaxies=samples.instance_drawn_randomly_from_pdf().galaxies,
                )
                for i in range(total_samples)
            ]

        func = partial(func_gen, total_samples=total_samples)

        return self.aggregator.map(func=func)
