from os import path

import autogalaxy as ag
import pytest
from test_autogalaxy.mock import mock_pipeline

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


def test__masked_imaging__settings_inputs_are_used_in_masked_imaging(
    interferometer_7, mask_7x7
):

    phase_interferometer_7 = ag.PhaseInterferometer(
        phase_name="phase_interferometer_7",
        settings=ag.PhaseSettingsInterferometer(
            grid_class=ag.Grid,
            grid_inversion_class=ag.Grid,
            sub_size=3,
            signal_to_noise_limit=1.0,
            bin_up_factor=2,
            inversion_pixel_limit=100,
            primary_beam_shape_2d=(3, 3),
        ),
        search=mock_pipeline.MockSearch(),
        real_space_mask=mask_7x7,
    )

    assert phase_interferometer_7.meta_dataset.settings.sub_size == 3
    assert phase_interferometer_7.meta_dataset.settings.signal_to_noise_limit == 1.0
    assert phase_interferometer_7.meta_dataset.settings.bin_up_factor == 2
    assert phase_interferometer_7.meta_dataset.settings.inversion_pixel_limit == 100
    assert phase_interferometer_7.meta_dataset.settings.primary_beam_shape_2d == (3, 3)

    analysis = phase_interferometer_7.make_analysis(
        dataset=interferometer_7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    assert isinstance(analysis.masked_dataset.grid, ag.Grid)
    assert isinstance(analysis.masked_dataset.grid_inversion, ag.Grid)
    assert isinstance(analysis.masked_dataset.transformer, ag.TransformerNUFFT)

    phase_interferometer_7 = ag.PhaseInterferometer(
        phase_name="phase_interferometer_7",
        settings=ag.PhaseSettingsInterferometer(
            grid_class=ag.GridIterate,
            sub_size=3,
            fractional_accuracy=0.99,
            sub_steps=[2],
            transformer_class=ag.TransformerDFT,
        ),
        search=mock_pipeline.MockSearch(),
        real_space_mask=mask_7x7,
    )

    analysis = phase_interferometer_7.make_analysis(
        dataset=interferometer_7, mask=mask_7x7, results=mock_pipeline.MockResults()
    )

    assert isinstance(analysis.masked_dataset.grid, ag.GridIterate)
    assert analysis.masked_dataset.grid.sub_size == 1
    assert analysis.masked_dataset.grid.fractional_accuracy == 0.99
    assert analysis.masked_dataset.grid.sub_steps == [2]
    assert isinstance(analysis.masked_dataset.transformer, ag.TransformerDFT)
