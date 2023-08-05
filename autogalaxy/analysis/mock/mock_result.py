from __future__ import annotations
from typing import TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from autogalaxy import mock

import autofit as af
import autogalaxy as ag


class MockResult(af.m.MockResult):
    def __init__(
        self,
        samples: mock.MockSamples = None,
        instance: af.Instance = None,
        model: af.Model = None,
        analysis: mock.MockAnalysis = None,
        search: af.mock.MockSearch = None,
        adapt_galaxy_image_path_dict: Dict[ag.Galaxy, ag.Array2D] = None,
        adapt_model_image: ag.Array2D = None,
    ):
        super().__init__(
            samples=samples,
            instance=instance,
            model=model,
            analysis=analysis,
            search=search,
        )

        self.adapt_galaxy_image_path_dict = adapt_galaxy_image_path_dict
        self.adapt_model_image = adapt_model_image

    @property
    def last(self):
        return self
