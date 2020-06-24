import autofit as af
import autogalaxy as ag
from test_autogalaxy.integration.tests.imaging import runner

test_type = "galaxy_x1"
test_name = "galaxy_x2__sersics__separate"
data_name = "galaxy_x2__sersics"
instrument = "vro"


def make_pipeline(name, folders, search=af.DynestyStatic()):

    bulge_0 = af.PriorModel(ag.lp.EllipticalSersic)

    bulge_0.centre_0 = -1.0
    bulge_0.centre_1 = -1.0

    phase1 = ag.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(galaxy_0=ag.GalaxyModel(redshift=0.5, bulge=bulge_0)),
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 40
    phase1.search.facc = 0.8

    bulge_1 = af.PriorModel(ag.lp.EllipticalSersic)

    bulge_1.centre_0 = 1.0
    bulge_1.centre_1 = 1.0

    phase2 = ag.PhaseImaging(
        phase_name="phase_2",
        folders=folders,
        galaxies=dict(
            galaxy_0=phase1.result.instance.galaxies.galaxy_0,
            galaxy_1=ag.GalaxyModel(redshift=0.5, bulge=bulge_1),
        ),
        search=search,
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 40
    phase2.search.facc = 0.8

    phase3 = ag.PhaseImaging(
        phase_name="phase_3",
        folders=folders,
        galaxies=dict(
            galaxy_0=phase1.result.model.galaxies.galaxy_0,
            galaxy_1=phase2.result.model.galaxies.galaxy_1,
        ),
        search=search,
    )

    phase3.search.const_efficiency_mode = True
    phase3.search.n_live_points = 60
    phase3.search.facc = 0.8

    return ag.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
