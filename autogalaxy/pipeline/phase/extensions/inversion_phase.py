import autofit as af
from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg
from autogalaxy.hyper import hyper_data as hd
from autogalaxy.pipeline.phase import abstract
from autogalaxy.pipeline.phase.imaging.phase import PhaseImaging

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
        phase = super().make_hyper_phase()

        return phase

    def make_model(self, instance):
        return instance.as_model(self.model_classes)

    def run_hyper(self, dataset, info=None, results=None, **kwargs):
        """
        Run the phase, overriding the search's model instance with one created to
        only fit pixelization hyperparameters.
        """

        self.results = results or af.ResultsCollection()

        phase = self.make_hyper_phase()
        phase.model = self.make_model(results.last.instance)

        return phase.run(dataset, mask=results.last.mask, results=results)


class InversionPhase(ModelFixingHyperPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(
        self,
        phase: abstract.AbstractPhase,
        search,
        model_classes=(pix.Pixelization, reg.Regularization),
    ):
        super().__init__(
            phase=phase,
            search=search,
            model_classes=model_classes,
            hyper_name="inversion",
        )


class InversionBackgroundSkyPhase(InversionPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging, search):
        super().__init__(
            phase=phase,
            search=search,
            model_classes=(pix.Pixelization, reg.Regularization, hd.HyperImageSky),
        )


class InversionBackgroundNoisePhase(InversionPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging, search):
        super().__init__(
            phase=phase,
            search=search,
            model_classes=(
                pix.Pixelization,
                reg.Regularization,
                hd.HyperBackgroundNoise,
            ),
        )


class InversionBackgroundBothPhase(InversionPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging, search):
        super().__init__(
            phase=phase,
            search=search,
            model_classes=(
                pix.Pixelization,
                reg.Regularization,
                hd.HyperImageSky,
                hd.HyperBackgroundNoise,
            ),
        )
