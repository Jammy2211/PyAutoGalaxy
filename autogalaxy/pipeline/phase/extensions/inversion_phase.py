import autofit as af
from autogalaxy.pipeline.phase import abstract

from .hyper_phase import HyperPhase


# noinspection PyAbstractClass
class ModelFixingHyperPhase(HyperPhase):
    def __init__(
        self,
        phase: abstract.AbstractPhase,
        search,
        hyper_name: str,
        model_classes=tuple(),
    ):

        super().__init__(phase=phase, search=search, hyper_name=hyper_name)

        self.model_classes = model_classes

    def make_hyper_phase(self):
        return super().make_hyper_phase()

    def make_model(self, instance):
        return instance.as_model(self.model_classes)

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


class InversionPhase(ModelFixingHyperPhase):
    def __init__(self, phase: abstract.AbstractPhase, search, model_classes=tuple()):
        super().__init__(
            phase=phase,
            search=search,
            model_classes=model_classes,
            hyper_name="inversion",
        )
