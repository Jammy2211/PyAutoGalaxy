import autofit as af
import autogalaxy as ag


def test__adapt_search():
    setup = ag.SetupAdapt(search_pix_cls=None, search_pix_dict=None)

    assert setup.search_pix_cls == af.DynestyStatic
    assert setup.search_pix_dict == {
        "nlive": 100,
    }

    setup = ag.SetupAdapt(
        search_pix_cls=af.DynestyDynamic,
        search_pix_dict={"hello": "there"},
    )
    assert setup.search_pix_cls == af.DynestyDynamic
    assert setup.search_pix_dict == {"hello": "there"}
