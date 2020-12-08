from os import path
import autofit as af
from autofit.tools.phase import Dataset
from autogalaxy.pipeline.phase import abstract


class HyperPhase:
    def __init__(self, phase: abstract.AbstractPhase, hyper_search, model_classes=tuple()):
        """
        Abstract HyperPhase. Wraps a phase, performing that phase before performing the action
        specified by the run_hyper.

        Parameters
        ----------
        phase
            A phase
        """
        self.phase = phase
        self.hyper_search = hyper_search
        self.model_classes = model_classes

    @property
    def hyper_name(self):
        return "hyper"

    def make_model(self, instance):
        return instance.as_model(self.model_classes)

    def make_hyper_phase(self) -> abstract.AbstractPhase:
        """
        Returns
        -------
        hyper_phase
            A copy of the original phase with a modified name and path
        """

        self.phase.search = self.hyper_search.copy_with_name_extension(
            extension=self.phase.name, path_prefix=self.phase.paths.path_prefix
        )
        self.phase.hyper_name = self.hyper_name
        return self.phase

    def run(
        self,
        dataset: Dataset,
        results: af.ResultsCollection,
        info=None,
        pickle_files=None,
        **kwargs,
    ) -> af.Result:
        """
        Run the hyper phase and then the hyper_galaxies phase.

        Parameters
        ----------
        dataset
            Data
        results
            Results from previous phases.
        kwargs

        Returns
        -------
        result
            The result of the phase, with a hyper_galaxies result attached as an attribute with the hyper_name of this
            phase.
        """
        result = self.phase.run(
            dataset, results=results, info=info, pickle_files=pickle_files, **kwargs
        )
        results.add(self.phase.paths.name, result)
        hyper_result = self.run_hyper(
            dataset=dataset,
            results=results,
            info=info,
            pickle_files=pickle_files,
            **kwargs,
        )
        setattr(result, self.hyper_name, hyper_result)
        return result

    def run_hyper(
        self,
        dataset,
        results: af.ResultsCollection,
        info=None,
        pickle_files=None,
        **kwargs
    ):
        """
        Run the phase, overriding the search's model instance with one created to
        only fit pixelization hyperparameters.
        """

        self.results = results

        phase = self.make_hyper_phase()
        phase.model = self.make_model(results.last.instance)

        return phase.run(
            dataset,
            mask=results.last.mask,
            results=results,
            info=info,
            pickle_files=pickle_files,
        )

    def __getattr__(self, item):
        return getattr(self.phase, item)
