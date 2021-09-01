import autofit as af
import autogalaxy as ag


class TestSetupHyper:
    def test__hyper_search(self):

        setup = ag.SetupHyper(search_inversion_cls=None, search_inversion_dict=None)
        assert setup.search_inversion_cls == af.DynestyStatic
        assert setup.search_inversion_dict == {
            "nlive": 50,
            "sample": "rstagger",
            "dlogz": 10,
        }

        setup = ag.SetupHyper(
            search_inversion_cls=af.DynestyDynamic,
            search_inversion_dict={"hello": "there"},
        )
        assert setup.search_inversion_cls == af.DynestyDynamic
        assert setup.search_inversion_dict == {"hello": "there"}

        setup = ag.SetupHyper(search_noise_cls=None, search_noise_dict=None)
        assert setup.search_noise_cls == af.DynestyStatic
        assert setup.search_noise_dict == {"nlive": 50, "sample": "rwalk"}

        setup = ag.SetupHyper(
            search_noise_cls=af.DynestyDynamic, search_noise_dict={"hello": "there"}
        )
        assert setup.search_noise_cls == af.DynestyDynamic
        assert setup.search_noise_dict == {"hello": "there"}
