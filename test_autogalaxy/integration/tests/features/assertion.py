import autofit as af
import autogalaxy as ag
from test_autogalaxy.integration.tests.imaging import runner

test_type = "features"
test_name = "assertion"
data_name = "galaxy_x1__dev_vaucouleurs"
instrument = "vro"


def make_pipeline(name, folders, search=af.DynestyStatic()):

    sersic = af.PriorModel(ag.lp.EllipticalSersic)

    # This will lead to pretty weird results

    sersic.add_assertion(sersic.axis_ratio > sersic.intensity)

    phase1 = ag.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(galaxy=ag.GalaxyModel(redshift=0.5, sersic=sersic)),
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 40
    phase1.search.facc = 0.8

    # TODO : And even with them not causing errors above, the promise doesnt work.

    phase2 = ag.PhaseImaging(
        phase_name="phase_2",
        folders=folders,
        galaxies=dict(galaxy=phase1.result.model.galaxies.light),
        search=search,
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 40
    phase2.search.facc = 0.8

    return ag.PipelineDataset(name, phase1, phase2)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
