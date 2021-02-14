from os import path

import autogalaxy as ag
import pytest
from autogalaxy.fit.fit import FitInterferometer
from autogalaxy.mock import mock

pytestmark = pytest.mark.filterwarnings(
    "ignore:Using a non-tuple sequence for multidimensional indexing is deprecated; use `arr[tuple(seq)]` instead of "
    "`arr[seq]`. In the future this will be interpreted as an arrays index, `arr[np.arrays(seq)]`, which will result "
    "either in an error or a different result."
)

directory = path.dirname(path.realpath(__file__))


class TestFit:
    def test__fit_using_interferometer(
        self, interferometer_7, mask_7x7, visibilities_mask_7, samples_with_result
    ):

        phase_interferometer_7 = ag.PhaseInterferometer(
            galaxies=dict(
                galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic),
                source=ag.GalaxyModel(redshift=1.0, light=ag.lp.EllipticalSersic),
            ),
            search=mock.MockSearch(samples=samples_with_result, name="test_phase"),
            real_space_mask=mask_7x7,
        )

        result = phase_interferometer_7.run(
            dataset=interferometer_7,
            mask=visibilities_mask_7,
            results=mock.MockResults(),
        )
        assert isinstance(result.instance.galaxies[0], ag.Galaxy)
        assert isinstance(result.instance.galaxies[0], ag.Galaxy)

    def test__fit_figure_of_merit__matches_correct_fit_given_galaxy_profiles(
        self, interferometer_7, mask_7x7, visibilities_mask_7
    ):
        galalxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=0.1))

        phase_interferometer_7 = ag.PhaseInterferometer(
            galaxies=dict(galaxy=galalxy),
            settings=ag.SettingsPhaseInterferometer(
                settings_masked_interferometer=ag.SettingsMaskedInterferometer(
                    sub_size=2
                )
            ),
            search=mock.MockSearch(name="test_phase"),
            real_space_mask=mask_7x7,
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=visibilities_mask_7,
            results=mock.MockResults(),
        )
        instance = phase_interferometer_7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        masked_interferometer = ag.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7,
            real_space_mask=mask_7x7,
        )
        plane = analysis.plane_for_instance(instance=instance)

        fit = ag.FitInterferometer(
            masked_interferometer=masked_interferometer, plane=plane
        )

        assert fit.log_likelihood == fit_figure_of_merit

    def test__fit_figure_of_merit__includes_hyper_image_and_noise__matches_fit(
        self, interferometer_7, mask_7x7, visibilities_mask_7
    ):
        hyper_background_noise = ag.hyper_data.HyperBackgroundNoise(noise_scale=1.0)

        galalxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=0.1))

        phase_interferometer_7 = ag.PhaseInterferometer(
            galaxies=dict(galaxy=galalxy),
            hyper_background_noise=hyper_background_noise,
            settings=ag.SettingsPhaseInterferometer(
                settings_masked_interferometer=ag.SettingsMaskedInterferometer(
                    sub_size=4
                )
            ),
            search=mock.MockSearch(name="test_phase"),
            real_space_mask=mask_7x7,
        )

        analysis = phase_interferometer_7.make_analysis(
            dataset=interferometer_7,
            mask=visibilities_mask_7,
            results=mock.MockResults(),
        )
        instance = phase_interferometer_7.model.instance_from_unit_vector([])
        fit_figure_of_merit = analysis.log_likelihood_function(instance=instance)

        assert analysis.masked_interferometer.real_space_mask.sub_size == 4

        masked_interferometer = ag.MaskedInterferometer(
            interferometer=interferometer_7,
            visibilities_mask=visibilities_mask_7,
            real_space_mask=mask_7x7,
            settings=ag.SettingsMaskedInterferometer(sub_size=4),
        )
        plane = analysis.plane_for_instance(instance=instance)
        fit = FitInterferometer(
            masked_interferometer=masked_interferometer,
            plane=plane,
            hyper_background_noise=hyper_background_noise,
        )

        assert fit.log_likelihood == fit_figure_of_merit
