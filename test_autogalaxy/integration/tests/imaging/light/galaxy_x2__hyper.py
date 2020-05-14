import autofit as af
import autogalaxy as ag
from test_autogalaxy.integration.tests.imaging import runner

test_type = "galaxy_x1"
test_name = "galaxy_x2__sersics__hyper"
data_label = "galaxy_x2__sersics"
instrument = "vro"


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):

    bulge_0 = af.PriorModel(ag.lp.EllipticalSersic)
    bulge_1 = af.PriorModel(ag.lp.EllipticalSersic)

    bulge_0.centre_0 = -1.0
    bulge_0.centre_1 = -1.0
    bulge_1.centre_0 = 1.0
    bulge_1.centre_1 = 1.0

    phase1 = ag.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            galaxy_0=ag.GalaxyModel(redshift=0.5, bulge=ag.lp.EllipticalSersic),
            galaxy_1=ag.GalaxyModel(redshift=0.5, bulge=ag.lp.EllipticalSersic),
        ),
        non_linear_class=non_linear_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    phase1 = phase1.extend_with_multiple_hyper_phases(hyper_galaxy=True)

    phase2 = ag.PhaseImaging(
        phase_name="phase_2",
        phase_folders=phase_folders,
        galaxies=dict(
            galaxy_0=ag.GalaxyModel(
                redshift=0.5,
                bulge=phase1.result.model.galaxies.galaxy_0.bulge,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.galaxy_0.hyper_galaxy,
            ),
            galaxy_1=ag.GalaxyModel(
                redshift=0.5,
                bulge=phase1.result.model.galaxies.galaxy_1.bulge,
                hyper_galaxy=phase1.result.hyper_combined.instance.galaxies.galaxy_1.hyper_galaxy,
            ),
        ),
        non_linear_class=non_linear_class,
    )

    phase2.optimizer.const_efficiency_mode = True
    phase2.optimizer.n_live_points = 40
    phase2.optimizer.sampling_efficiency = 0.8

    return ag.PipelineDataset(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
