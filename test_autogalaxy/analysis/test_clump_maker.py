import autofit as af
import autogalaxy as ag


def test__clump_dict_from():

    centres = ag.Grid2DIrregular(grid=[(1.0, 1.0)])

    clump_maker = ag.ClumpMaker(redshift=0.5, light_cls=ag.lp.SphSersic)

    clump_dict = clump_maker.clump_dict_from(centres=centres)

    assert clump_dict["clump_0"].redshift == 0.5
    assert clump_dict["clump_0"].light.centre == (1.0, 1.0)
    assert isinstance(clump_dict["clump_0"].light.intensity, af.LogUniformPrior)
    assert clump_dict["clump_0"].mass == None

    clump_maker = ag.ClumpMaker(redshift=0.5, mass_cls=ag.mp.SphIsothermal)

    clump_dict = clump_maker.clump_dict_from(centres=centres)

    assert clump_dict["clump_0"].redshift == 0.5
    assert clump_dict["clump_0"].mass.centre == (1.0, 1.0)
    assert isinstance(clump_dict["clump_0"].mass.einstein_radius, af.UniformPrior)
    assert clump_dict["clump_0"].light == None

    clump_maker = ag.ClumpMaker(
        redshift=0.5, light_cls=ag.lp.SphSersic, mass_cls=ag.mp.SphIsothermal
    )

    clump_dict = clump_maker.clump_dict_from(centres=centres)

    assert clump_dict["clump_0"].redshift == 0.5
    assert clump_dict["clump_0"].light.centre == (1.0, 1.0)
    assert isinstance(clump_dict["clump_0"].light.intensity, af.LogUniformPrior)
    assert isinstance(clump_dict["clump_0"].mass.einstein_radius, af.UniformPrior)
