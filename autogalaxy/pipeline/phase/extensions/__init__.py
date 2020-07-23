import autofit as af
from autogalaxy.pipeline.phase.imaging import PhaseImaging
from .hyper_galaxy_phase import HyperGalaxyPhase
from .hyper_phase import HyperPhase
from .inversion_phase import ModelFixingHyperPhase, InversionPhase


class CombinedHyperPhase(HyperPhase):
    def __init__(
        self,
        phase: PhaseImaging,
        search: af.NonLinearSearch,
        hyper_phases: (HyperPhase,) = tuple(),
    ):
        """
        A hyper_combined hyper_galaxies phase that can run zero or more other hyper_galaxies phases after the initial phase is
        run.

        Parameters
        ----------
        phase : phase_PhaseImaging
            The phase wrapped by this hyper_galaxies phase
        hyper_phases
            The classes of hyper_galaxies phases to be run following the initial phase
        """
        super().__init__(phase=phase, search=search, hyper_name="hyper_combined")
        self.hyper_phases = hyper_phases

    @property
    def phase_names(self) -> [str]:
        """
        The names of phases included in this hyper_combined phase
        """
        return [phase.hyper_name for phase in self.hyper_phases]

    def run(
        self,
        dataset,
        mask,
        results: af.ResultsCollection,
        info=None,
        pickle_files=None,
        **kwargs
    ) -> af.Result:
        """
        Run the phase followed by the hyper_galaxies phases. Each result of a hyper_galaxies phase is attached to the
        overall result object by the hyper_name of that phase.

        Finally, a phase in run with all of the model results from all the individual hyper_galaxies phases.

        Parameters
        ----------
        mask
        data
            the dataset
        results
            Results from previous phases
        kwargs

        Returns
        -------
        result
            The result of the phase, with hyper_galaxies results attached by associated hyper_galaxies names
        """
        results = results.copy()

        result = self.phase.run(
            dataset=dataset,
            mask=mask,
            results=results,
            info=info,
            pickle_files=pickle_files,
            **kwargs
        )
        results.add(self.phase.paths.name, result)

        for phase in self.hyper_phases:
            hyper_result = phase.run_hyper(
                dataset=dataset,
                results=results,
                info=info,
                pickle_files=pickle_files,
                **kwargs
            )
            setattr(result, phase.hyper_name, hyper_result)

        setattr(
            result, self.hyper_name, self.run_hyper(dataset=dataset, results=results)
        )
        return result

    def combine_models(self, result) -> af.ModelMapper:
        """
        Combine the model objects from all previous results in this hyper_combined hyper_galaxies phase.

        Iterates through the hyper_galaxies names of the included hyper_galaxies phases, extracting a result
        for each name and adding the model of that result to a new model.

        Parameters
        ----------
        result
            The last result (with attribute results associated with phases in this phase)

        Returns
        -------
        combined_model
            A model object including all models from results in this phase.
        """
        model = af.ModelMapper()
        for name in self.phase_names:
            model += getattr(result, name).model
        return model

    def run_hyper(
        self,
        dataset,
        results: af.ResultsCollection,
        info=None,
        pickle_files=None,
        **kwargs
    ) -> af.Result:

        phase = self.make_hyper_phase()
        phase.model = self.combine_models(results.last)

        return phase.run(
            dataset,
            mask=results.last.mask,
            results=results,
            info=info,
            pickle_files=pickle_files,
        )
