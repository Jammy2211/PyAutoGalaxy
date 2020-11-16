import autofit as af
import autogalaxy as ag
from test_autogalaxy.integration.tests.interferometer import runner

test_type = "galaxy_x1"
test_name = "galaxy_light"
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

    return ag.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
