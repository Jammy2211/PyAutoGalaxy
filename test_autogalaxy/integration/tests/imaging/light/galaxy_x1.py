import autofit as af
import autogalaxy as ag
from test_autogalaxy.integration.tests.imaging import runner

test_type = "light"
test_name = "galaxy_x1"
data_name = "galaxy_x1__dev_vaucouleurs"
instrument = "vro"


def make_pipeline(name, folders, search=af.DynestyStatic()):
    phase1 = ag.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, sersic=ag.lp.EllipticalSersic)
        ),
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 40
    phase1.search.facc = 0.8

    return ag.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
