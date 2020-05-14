import autofit as af
import autogalaxy as ag
from test_autogalaxy.integration.tests.imaging import runner

test_type = "phase_features"
test_name = "bin_up_factor"
data_label = "galaxy_x1__dev_vaucouleurs"
instrument = "vro"


def make_pipeline(name, phase_folders, non_linear_class=af.MultiNest):

    phase1 = ag.PhaseImaging(
        phase_name="phase_1",
        phase_folders=phase_folders,
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, sersic=ag.lp.EllipticalSersic)
        ),
        bin_up_factor=2,
        non_linear_class=non_linear_class,
    )

    phase1.optimizer.const_efficiency_mode = True
    phase1.optimizer.n_live_points = 40
    phase1.optimizer.sampling_efficiency = 0.8

    return ag.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
