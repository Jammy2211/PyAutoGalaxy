import autogalaxy as ag


class TestMassLightRelation:
    def test__einstein_radius_from(self):

        mass_light_relation = ag.sr.MassLightRelation(
            gradient=1.0, denominator=1.0, power=1.0
        )

        einstein_radius = mass_light_relation.einstein_radius_from(luminosity=1.0)

        assert einstein_radius == 1.0

        einstein_radius = mass_light_relation.einstein_radius_from(luminosity=2.0)

        assert einstein_radius == 2.0

        mass_light_relation = ag.sr.MassLightRelation(
            gradient=2.0, denominator=1.0, power=1.0
        )

        einstein_radius = mass_light_relation.einstein_radius_from(luminosity=2.0)

        assert einstein_radius == 4.0

        mass_light_relation = ag.sr.MassLightRelation(
            gradient=2.0, denominator=0.5, power=1.0
        )

        einstein_radius = mass_light_relation.einstein_radius_from(luminosity=4.0)

        assert einstein_radius == 16.0

        mass_light_relation = ag.sr.MassLightRelation(
            gradient=2.0, denominator=0.5, power=2.0
        )

        einstein_radius = mass_light_relation.einstein_radius_from(luminosity=4.0)

        assert einstein_radius == 128.0


class TestIsothermalMLR:
    def test__setup_correctly_from_luminosity(self):

        relation = ag.sr.MassLightRelation(gradient=2.0, denominator=0.5, power=2.0)

        sis = ag.sr.SphIsothermalMLR(
            relation=relation, luminosity=4.0, centre=(1.0, 1.0)
        )

        assert sis.relation == relation
        assert sis.luminosity == 4.0
        assert sis.centre == (1.0, 1.0)
        assert sis.einstein_radius == 128.0

        relation = ag.sr.MassLightRelation(gradient=2.0, denominator=0.5, power=2.0)

        sie = ag.sr.EllIsothermalMLR(
            relation=relation,
            luminosity=4.0,
            elliptical_comps=(0.5, 0.5),
            centre=(1.0, 1.0),
        )

        assert sie.relation == relation
        assert sie.luminosity == 4.0
        assert sie.centre == (1.0, 1.0)
        assert sie.elliptical_comps == (0.5, 0.5)
        assert sie.einstein_radius == 128.0
