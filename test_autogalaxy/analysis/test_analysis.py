import pytest

import autofit as af
import autogalaxy as ag
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)


class TestAnalysisImaging:

    # noinspection PyTypeChecker
    def test__mock_search_runs_without_exception(self, masked_imaging_7x7):

        search = mock.MockSearch(name="name")

        model = af.CollectionPriorModel(
            galaxies=af.CollectionPriorModel(
                galaxy=ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalLightProfile)
            )
        )

        analysis = ag.AnalysisImaging(dataset=masked_imaging_7x7)

        result = search.fit(model=model, analysis=analysis)

        assert result is not None
