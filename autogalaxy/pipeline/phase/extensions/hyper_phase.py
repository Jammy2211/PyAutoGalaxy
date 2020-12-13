from autoconf import conf
import autofit as af
from autogalaxy.galaxy import galaxy as g
from autofit.tools.phase import Dataset
from autogalaxy.pipeline.phase import abstract
import numpy as np


class HyperPhase:
    def __init__(
        self,
        phase: abstract.AbstractPhase,
        hyper_search,
        model_classes=tuple(),
        hyper_image_sky=None,
        hyper_background_noise=None,
        hyper_galaxy_names=None,
    ):
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
        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise
        self.hyper_galaxy_names = hyper_galaxy_names

    @property
    def hyper_name(self):
        return "hyper"

    def make_model(self, instance):

        model = instance.as_model(self.model_classes)
        model.hyper_image_sky = self.hyper_image_sky
        model.hyper_background_noise = self.hyper_background_noise

        return model

    def add_hyper_galaxies_to_model(
        self, model, path_galaxy_tuples, hyper_galaxy_image_path_dict
    ):

        for path_galaxy, galaxy in path_galaxy_tuples:
            if path_galaxy[-1] in self.hyper_galaxy_names:
                if not np.all(hyper_galaxy_image_path_dict[path_galaxy] == 0):

                    if "source" in path_galaxy[-1]:
                        setattr(
                            model.galaxies.source,
                            "hyper_galaxy",
                            af.PriorModel(g.HyperGalaxy),
                        )
                    elif "lens" in path_galaxy[-1]:
                        setattr(
                            model.galaxies.lens,
                            "hyper_galaxy",
                            af.PriorModel(g.HyperGalaxy),
                        )

        return model

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
        **kwargs,
    ):
        """
        Run the phase, overriding the search's model instance with one created to
        only fit pixelization hyperparameters.
        """

        self.results = results

        phase = self.make_hyper_phase()
        model = self.make_model(instance=results.last.instance)

        if self.hyper_galaxy_names is not None:
            model = self.add_hyper_galaxies_to_model(
                model=model,
                path_galaxy_tuples=results.last.path_galaxy_tuples,
                hyper_galaxy_image_path_dict=results.last.hyper_galaxy_image_path_dict,
            )

        phase.model = model

        return phase.run(
            dataset,
            mask=results.last.mask,
            results=results,
            info=info,
            pickle_files=pickle_files,
        )

    def __getattr__(self, item):
        return getattr(self.phase, item)
