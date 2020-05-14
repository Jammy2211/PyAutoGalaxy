import autofit as af
import autogalaxy as ag
from test_autogalaxy.integration.tests.imaging import runner

test_type = "bulge"
test_name = "galaxy_x2"
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
            galaxy_0=ag.GalaxyModel(redshift=0.5, bulge=bulge_0),
            galaxy_1=ag.GalaxyModel(redshift=0.5, bulge=bulge_1),
        ),
        non_linear_class=non_linear_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return ag.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
