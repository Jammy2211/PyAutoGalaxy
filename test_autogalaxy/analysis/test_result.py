import pytest
import numpy as np

import autofit as af
import autogalaxy as ag
from autogalaxy.analysis import result as res
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)


class TestResultAbstract:
    def test__result_contains_instance_with_galaxies(
        self, masked_imaging_7x7, samples_with_result
    ):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(
                galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
                source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
            )
        )

        search = mock.MockSearch(samples=samples_with_result, name="test_phase")

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result.instance.galaxies[0], ag.Galaxy)
        assert isinstance(result.instance.galaxies[1], ag.Galaxy)

    def test__results_of_phase_are_available_as_properties(self, masked_imaging_7x7):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(
                galaxy=ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalLightProfile)
            )
        )

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

        search = mock.MockSearch(name="test_phase_2")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.Result)

    def test__max_log_likelihood_plane_available_as_result(self, masked_imaging_7x7):

        galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0))
        galaxy_1 = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=2.0))

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(galaxy_0=galaxy_0, galaxy_1=galaxy_1)
        )

        max_log_likelihood_plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

        search = mock.MockSearch(
            name="test_phase",
            samples=mock.MockSamples(
                max_log_likelihood_instance=max_log_likelihood_plane
            ),
        )

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result.max_log_likelihood_plane, ag.Plane)
        assert result.max_log_likelihood_plane.galaxies[0].light.intensity == 1.0
        assert result.max_log_likelihood_plane.galaxies[1].light.intensity == 2.0


class TestResultDataset:
    def test__results_of_include_masked_dataset_and_mask(
        self, masked_imaging_7x7, samples_with_result
    ):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(
                galaxy=ag.Galaxy(
                    redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0)
                )
            )
        )

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

        search = mock.MockSearch("test_phase_2", samples=samples_with_result)

        result = search.fit(model=model, analysis=analysis)

        assert (result.mask == masked_imaging_7x7.mask).all()
        assert result.dataset == masked_imaging_7x7

    def test__results_of_phase_include_pixelization__available_as_property(
        self, masked_imaging_7x7
    ):
        source = ag.Galaxy(
            redshift=1.0,
            pixelization=ag.pix.VoronoiMagnification(shape=(2, 3)),
            regularization=ag.reg.Constant(),
        )

        max_log_likelihood_plane = ag.Plane(galaxies=[source])

        samples = mock.MockSamples(max_log_likelihood_instance=max_log_likelihood_plane)

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(galaxy=ag.Galaxy(redshift=0.5))
        )

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

        search = mock.MockSearch("test_phase_2", samples=samples)

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result.pixelization, ag.pix.VoronoiMagnification)
        assert result.pixelization.shape == (2, 3)

        source = ag.Galaxy(
            redshift=1.0,
            pixelization=ag.pix.VoronoiBrightnessImage(pixels=6),
            regularization=ag.reg.Constant(),
        )

        source.hyper_galaxy_image = np.ones(9)

        max_log_likelihood_plane = ag.Plane(galaxies=[source])

        samples = mock.MockSamples(max_log_likelihood_instance=max_log_likelihood_plane)

        model = af.CollectionPriorModel(galaxies=af.CollectionPriorModel(source=source))

        model.galaxies.source.hyper_galaxy_image = np.ones(9)

        search = mock.MockSearch("test_phase_2", samples=samples)

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result.pixelization, ag.pix.VoronoiBrightnessImage)
        assert result.pixelization.pixels == 6


class TestResultImaging:
    def test__result_imaging_is_returned(self, masked_imaging_7x7):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(galaxy_0=ag.Galaxy(redshift=0.5))
        )

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

        search = mock.MockSearch(name="test_phase")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultImaging)

    def test___image_dict(self, masked_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.galaxy = ag.Galaxy(redshift=0.5)
        galaxies.source = ag.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        analysis = ag.AnalysisImaging(
            dataset=masked_imaging_7x7, results=mock.MockResults()
        )

        result = res.ResultImaging(
            samples=mock.MockSamples(max_log_likelihood_instance=instance),
            model=af.ModelMapper(),
            analysis=analysis,
            search=None,
        )

        image_dict = result.image_galaxy_dict
        assert isinstance(image_dict[("galaxies", "galaxy")], np.ndarray)
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

        result.instance.galaxies.light = ag.Galaxy(redshift=0.5)

        image_dict = result.image_galaxy_dict
        assert (image_dict[("galaxies", "galaxy")].native == np.zeros((7, 7))).all()
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)


class TestResultInterferometer:
    def test__result_interferometer_is_returned(self, masked_interferometer_7):

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(galaxy_0=ag.Galaxy(redshift=0.5))
        )

        analysis = ag.AnalysisInterferometer(dataset=masked_interferometer_7)

        search = mock.MockSearch(name="test_phase")

        result = search.fit(model=model, analysis=analysis)

        assert isinstance(result, res.ResultInterferometer)
