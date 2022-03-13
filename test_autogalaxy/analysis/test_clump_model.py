import autofit as af
import autogalaxy as ag


def test__clump_dict():

    centres = ag.Grid2DIrregular(grid=[(1.0, 1.0)])

    clump_model = ag.ClumpModel(
        redshift=0.5, centres=centres, light_cls=ag.lp.SphSersic
    )

    clump_dict = clump_model.clump_dict

    assert clump_dict["clump_0"].redshift == 0.5
    assert clump_dict["clump_0"].light.centre == (1.0, 1.0)
    assert isinstance(clump_dict["clump_0"].light.intensity, af.LogUniformPrior)
    assert clump_dict["clump_0"].mass == None

    clump_model = ag.ClumpModel(
        redshift=0.5, centres=centres, mass_cls=ag.mp.SphIsothermal
    )

    clump_dict = clump_model.clump_dict

    assert clump_dict["clump_0"].redshift == 0.5
    assert clump_dict["clump_0"].mass.centre == (1.0, 1.0)
    assert isinstance(clump_dict["clump_0"].mass.einstein_radius, af.UniformPrior)
    assert clump_dict["clump_0"].light == None

    clump_model = ag.ClumpModel(
        redshift=0.5,
        centres=centres,
        light_cls=ag.lp.SphSersic,
        mass_cls=ag.mp.SphIsothermal,
    )

    clump_dict = clump_model.clump_dict

    assert clump_dict["clump_0"].redshift == 0.5
    assert clump_dict["clump_0"].light.centre == (1.0, 1.0)
    assert isinstance(clump_dict["clump_0"].light.intensity, af.LogUniformPrior)
    assert isinstance(clump_dict["clump_0"].mass.einstein_radius, af.UniformPrior)


def test__einstein_radius_max():

    centres = ag.Grid2DIrregular(grid=[(1.0, 1.0)])

    clump_model = ag.ClumpModel(
        redshift=0.5,
        centres=centres,
        mass_cls=ag.mp.SphIsothermal,
        einstein_radius_upper_limit=1.0,
    )

    clump_dict = clump_model.clump_dict

    assert isinstance(clump_dict["clump_0"].mass.einstein_radius, af.UniformPrior)
    assert clump_dict["clump_0"].mass.einstein_radius.upper_limit == 1.0

    assert clump_dict["clump_0"].redshift == 0.5
    assert clump_dict["clump_0"].mass.centre == (1.0, 1.0)
    assert clump_dict["clump_0"].light == None
