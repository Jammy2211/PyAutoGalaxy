import autofit as af
import autogalaxy as ag

from autogalaxy.analysis import result as res


class TestResultAbstract:
    def test__result_contains_instance_with_galaxies(
        self, analysis_imaging_7x7, samples_with_result
    ):

        model = af.Collection(
            galaxies=af.Collection(
                galaxy=af.Model(ag.Galaxy, redshift=0.5, light=ag.lp.EllSersic),
                source=af.Model(ag.Galaxy, redshift=1.0, light=ag.lp.EllSersic),
            )
        )

        result = res.Result(
            samples=samples_with_result,
            analysis=analysis_imaging_7x7,
            model=model,
            search=None,
        )

        assert isinstance(result.instance.galaxies[0], ag.Galaxy)
        assert isinstance(result.instance.galaxies[1], ag.Galaxy)

    def test__max_log_likelihood_plane_available_as_result(self, analysis_imaging_7x7):

        galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=1.0))
        galaxy_1 = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=2.0))

        model = af.Collection(
            galaxies=af.Collection(galaxy_0=galaxy_0, galaxy_1=galaxy_1)
        )

        max_log_likelihood_plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

        search = ag.m.MockSearch(
            name="test_search",
            samples=ag.m.MockSamples(
                max_log_likelihood_instance=max_log_likelihood_plane
            ),
        )

        result = search.fit(model=model, analysis=analysis_imaging_7x7)

        assert isinstance(result.max_log_likelihood_plane, ag.Plane)
        assert result.max_log_likelihood_plane.galaxies[0].light.intensity == 1.0
        assert result.max_log_likelihood_plane.galaxies[1].light.intensity == 2.0


class TestResultDataset:
    def test__results_include_pixelization__available_as_property(
        self, analysis_imaging_7x7
    ):
        source = ag.Galaxy(
            redshift=1.0,
            pixelization=ag.m.MockPixelization(mapper=1),
            regularization=ag.m.MockRegularization(),
        )

        max_log_likelihood_plane = ag.Plane(galaxies=[source])

        samples = ag.m.MockSamples(max_log_likelihood_instance=max_log_likelihood_plane)

        result = res.ResultDataset(
            samples=samples, analysis=analysis_imaging_7x7, model=None, search=None
        )

        assert isinstance(result.pixelization_list[0], ag.m.MockPixelization)
        assert result.pixelization_list[0].mapper == 1
