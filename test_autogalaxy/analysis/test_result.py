import pytest

import autofit as af
import autogalaxy as ag
from autogalaxy.analysis import result as res
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)


class TestGeneric:
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


class TestPlane:
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
