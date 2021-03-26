import autofit as af
import autogalaxy as ag
from autogalaxy import exc

import pytest


class TestSetupHyper:
    def test__hyper_search(self):

        setup = ag.SetupHyper(search=None)
        assert setup.search.n_live_points == 50
        assert setup.search.evidence_tolerance == pytest.approx(0.059, 1.0e-4)

        setup = ag.SetupHyper(search=af.DynestyStatic(n_live_points=51))
        assert setup.search.n_live_points == 51

        setup = ag.SetupHyper(hyper_galaxies=True, evidence_tolerance=0.5)
        assert setup.search.evidence_tolerance == 0.5
        assert setup.search.evidence_tolerance == 0.5

        with pytest.raises(exc.PipelineException):
            ag.SetupHyper(
                search=af.DynestyStatic(n_live_points=51), evidence_tolerance=3.0
            )
