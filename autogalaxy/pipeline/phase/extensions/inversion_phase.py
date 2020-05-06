import autofit as af
from autogalaxy.hyper import hyper_data as hd
from autoarray.operators.inversion import pixelizations as pix
from autoarray.operators.inversion import regularization as reg
from autogalaxy.pipeline.phase import abstract
from autogalaxy.pipeline.phase.imaging.phase import PhaseImaging
from .hyper_phase import HyperPhase


# noinspection PyAbstractClass
class ModelFixingHyperPhase(HyperPhase):
    def __init__(
        self,
        phase: abstract.AbstractPhase,
        hyper_name: str,
        non_linear_class=af.MultiNest,
        model_classes=tuple(),
    ):
        super().__init__(
            phase=phase, hyper_name=hyper_name, non_linear_class=non_linear_class
        )

        self.model_classes = model_classes

    def make_hyper_phase(self):
        phase = super().make_hyper_phase()

        self.update_optimizer_with_config(
            optimizer=phase.optimizer, section="inversion"
        )

        return phase

    def make_model(self, instance):
        return instance.as_model(self.model_classes)

    def run_hyper(self, dataset, info=None, results=None, **kwargs):
        """
        Run the phase, overriding the optimizer's model instance with one created to
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
        model_classes=(pix.Pixelization, reg.Regularization),
        non_linear_class=af.MultiNest,
    ):
        super().__init__(
            phase=phase,
            model_classes=model_classes,
            non_linear_class=non_linear_class,
            hyper_name="inversion",
        )


class InversionBackgroundSkyPhase(InversionPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging, non_linear_class=af.MultiNest):
        super().__init__(
            phase=phase,
            model_classes=(pix.Pixelization, reg.Regularization, hd.HyperImageSky),
            non_linear_class=non_linear_class,
        )


class InversionBackgroundNoisePhase(InversionPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging, non_linear_class=af.MultiNest):
        super().__init__(
            phase=phase,
            model_classes=(
                pix.Pixelization,
                reg.Regularization,
                hd.HyperBackgroundNoise,
            ),
            non_linear_class=non_linear_class,
        )


class InversionBackgroundBothPhase(InversionPhase):
    """
    Phase that makes everything in the model from the previous phase equal to the
    corresponding value from the best fit except for models associated with
    pixelization
    """

    def __init__(self, phase: PhaseImaging, non_linear_class=af.MultiNest):
        super().__init__(
            phase=phase,
            model_classes=(
                pix.Pixelization,
                reg.Regularization,
                hd.HyperImageSky,
                hd.HyperBackgroundNoise,
            ),
            non_linear_class=non_linear_class,
        )
