import autofit as af
import autogalaxy as ag


class TestSetupHyper:
    def test__hyper_search(self):

        setup = ag.SetupHyper(search_cls=None, search_dict=None)
        assert setup.search_cls == af.DynestyStatic
        assert setup.search_dict == {"nlive": 50, "sample": "rstagger", "dlogz": 10}

        setup = ag.SetupHyper(search_cls=af.DynestyDynamic, search_dict={"hi": "there"})
        assert setup.search_cls == af.DynestyDynamic
        assert setup.search_dict == {"hi": "there"}
