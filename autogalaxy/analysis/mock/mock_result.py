from __future__ import annotations
from typing import TYPE_CHECKING, Dict, List

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
        mask: mock.MockMask = None,
        model_image: ag.Array2D = None,
        path_galaxy_tuples: List[(str, ag.Galaxy)] = None,
        adapt_galaxy_image_path_dict: Dict[ag.Galaxy, ag.Array2D] = None,
        adapt_model_image: ag.Array2D = None,
        pixelization: ag.Pixelization = None,
    ):
        super().__init__(
            samples=samples,
            instance=instance,
            model=model,
            analysis=analysis,
            search=search,
        )

        self.mask = mask
        self.adapt_galaxy_image_path_dict = adapt_galaxy_image_path_dict
        self.adapt_model_image = adapt_model_image
        self.path_galaxy_tuples = path_galaxy_tuples
        self.model_image = model_image
        self.unmasked_model_image = model_image
        self.pixelization = pixelization

        self.max_log_likelihood_plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)])

    @property
    def last(self):
        return self
