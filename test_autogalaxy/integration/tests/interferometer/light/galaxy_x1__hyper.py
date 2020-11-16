import autofit as af
import autogalaxy as ag
from test_autogalaxy.integration.tests.interferometer import runner

test_type = "galaxy_x1"
test_name = "galaxy_bulge__hyper_bg"
data_name = "galaxy_x1__dev_vaucouleurs"
instrument = "sma"


def make_pipeline(name, path_prefix, real_space_mask, search=af.DynestyStatic()):

    phase1 = ag.PhaseInterferometer(
        name="phase_1",
        path_prefix=path_prefix,
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, bulge=ag.lp.EllipticalSersic)
        ),
        real_space_mask=real_space_mask,
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 40
    phase1.search.facc = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=True,
        include_background_sky=True,
        include_background_noise=True,
    )

    phase2 = ag.PhaseInterferometer(
        name="phase_2",
        path_prefix=path_prefix,
        galaxies=dict(
            galaxy=ag.GalaxyModel(
                redshift=0.5,
                bulge=phase1.result.model.galaxies.light.bulge,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.light.hyper_galaxy,
            )
        ),
        hyper_background_noise=phase1.result.hyper_combined.instance.hyper_background_noise,
        real_space_mask=real_space_mask,
        search=search,
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 40
    phase2.search.facc = 0.8

    return ag.PipelineDataset(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
