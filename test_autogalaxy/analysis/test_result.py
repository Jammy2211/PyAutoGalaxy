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

        search = mock.MockSearch(
            name="test_search",
            samples=mock.MockSamples(
                max_log_likelihood_instance=max_log_likelihood_plane
            ),
        )

        result = search.fit(model=model, analysis=analysis_imaging_7x7)

        assert isinstance(result.max_log_likelihood_plane, ag.Plane)
        assert result.max_log_likelihood_plane.galaxies[0].light.intensity == 1.0
        assert result.max_log_likelihood_plane.galaxies[1].light.intensity == 2.0


class TestResultDataset:
    def test__results_include_masked_dataset_and_mask(
        self, analysis_imaging_7x7, masked_imaging_7x7, samples_with_result
    ):

        result = res.ResultDataset(
            samples=samples_with_result,
            analysis=analysis_imaging_7x7,
            model=None,
            search=None,
        )

        assert (result.mask == masked_imaging_7x7.mask).all()
        assert (result.dataset.image == masked_imaging_7x7.image).all()

    def test__results_include_pixelization__available_as_property(
        self, analysis_imaging_7x7
    ):
        source = ag.Galaxy(
            redshift=1.0,
            pixelization=ag.pix.VoronoiMagnification(shape=(2, 3)),
            regularization=ag.reg.Constant(),
        )

        max_log_likelihood_plane = ag.Plane(galaxies=[source])

        samples = mock.MockSamples(max_log_likelihood_instance=max_log_likelihood_plane)

        result = res.ResultDataset(
            samples=samples, analysis=analysis_imaging_7x7, model=None, search=None
        )

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

        result = res.ResultDataset(
            samples=samples, analysis=analysis_imaging_7x7, model=None, search=None
        )

        assert isinstance(result.pixelization, ag.pix.VoronoiBrightnessImage)
        assert result.pixelization.pixels == 6


class TestResultImaging:
    def test___image_dict(self, analysis_imaging_7x7):

        galaxies = af.ModelInstance()
        galaxies.galaxy = ag.Galaxy(redshift=0.5)
        galaxies.source = ag.Galaxy(redshift=1.0)

        instance = af.ModelInstance()
        instance.galaxies = galaxies

        samples = mock.MockSamples(max_log_likelihood_instance=instance)

        result = res.ResultImaging(
            samples=samples, analysis=analysis_imaging_7x7, model=None, search=None
        )

        image_dict = result.image_galaxy_dict
        assert isinstance(image_dict[("galaxies", "galaxy")], np.ndarray)
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

        result.instance.galaxies.light = ag.Galaxy(redshift=0.5)

        image_dict = result.image_galaxy_dict
        assert (image_dict[("galaxies", "galaxy")].native == np.zeros((7, 7))).all()
        assert isinstance(image_dict[("galaxies", "source")], np.ndarray)


class TestResultInterferometer:

    pass
