import autofit as af
import autogalaxy as ag
from autogalaxy import exc

import pytest


class TestSetupHyper:
    def test__hyper_search(self):

        setup = ag.SetupHyper(search=None)
        assert setup.search.config_dict_search["nlive"] == 150
        assert setup.search.config_dict_search["dlogz"] == None

        setup = ag.SetupHyper(search=af.DynestyStatic(nlive=51))
        assert setup.search.config_dict_search["nlive"] == 51

        setup = ag.SetupHyper(hyper_galaxies=True, dlogz=0.5)
        assert setup.search.config_dict_search["dlogz"] == 0.5

        with pytest.raises(exc.PipelineException):
            ag.SetupHyper(search=af.DynestyStatic(nlive=51), dlogz=3.0)
