from os import path

import autofit as af
import autogalaxy as ag

from autogalaxy.quantity.model.result import ResultQuantity

directory = path.dirname(path.realpath(__file__))


class TestAnalysisQuantity:
    def test__make_result__result_quantity_is_returned(
        self, dataset_quantity_7x7_array_2d
    ):

        model = af.Collection(galaxies=af.Collection(galaxy_0=ag.Galaxy(redshift=0.5)))

        analysis = ag.AnalysisQuantity(
            dataset=dataset_quantity_7x7_array_2d, func_str="convergence_2d_from"
        )

        search = ag.m.MockSearch(name="test_search")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, ResultQuantity)

    def test__figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, dataset_quantity_7x7_array_2d
    ):
        galaxy = ag.Galaxy(redshift=0.5, light=ag.mp.EllIsothermal(einstein_radius=1.0))

        model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

        analysis = ag.AnalysisQuantity(
            dataset=dataset_quantity_7x7_array_2d, func_str="convergence_2d_from"
        )

        instance = model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        plane = analysis.plane_via_instance_from(instance=instance)

        fit = ag.FitQuantity(
            dataset=dataset_quantity_7x7_array_2d,
            light_mass_obj=plane,
            func_str="convergence_2d_from",
        )

        assert fit.log_likelihood == fit_figure_of_merit

        fit = ag.FitQuantity(
            dataset=dataset_quantity_7x7_array_2d,
            light_mass_obj=plane,
            func_str="potential_2d_from",
        )

        assert fit.log_likelihood != fit_figure_of_merit
