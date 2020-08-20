import copy

import autofit as af
from autofit.tools.phase import Dataset
from autogalaxy.pipeline.phase import abstract


class HyperPhase:
    def __init__(self, phase: abstract.AbstractPhase, search, hyper_name: str):
        """
        Abstract HyperPhase. Wraps a phase, performing that phase before performing the action
        specified by the run_hyper.

        Parameters
        ----------
        phase
            A phase
        """
        self.phase = phase
        self.hyper_name = hyper_name
        self.search = search

    def run_hyper(self, *args, **kwargs) -> af.Result:
        """
        Run the hyper_galaxies phase.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------
        result
            The result of the hyper_galaxies phase.
        """
        raise NotImplementedError()

    def make_hyper_phase(self) -> abstract.AbstractPhase:
        """
        Returns
        -------
        hyper_phase
            A copy of the original phase with a modified name and path
        """

        phase = copy.deepcopy(self.phase)
        phase.search = self.search.copy_with_name_extension(
            extension=f"{self.phase.paths.path_prefix}/{self.phase.phase_name}"
        )
        phase.hyper_name = self.hyper_name
        return phase

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

    def __getattr__(self, item):
        return getattr(self.phase, item)
