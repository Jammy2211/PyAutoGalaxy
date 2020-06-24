import autofit as af
import autogalaxy as ag
from test_autogalaxy.integration.tests.imaging import runner

test_type = "features"
test_name = "grid_search"
data_name = "galaxy_x1__dev_vaucouleurs"
instrument = "vro"


def make_pipeline(name, folders, search=af.DynestyStatic()):

    bulge = af.PriorModel(ag.lp.EllipticalDevVaucouleurs)

    bulge.centre_0 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    bulge.centre_1 = af.UniformPrior(lower_limit=-0.01, upper_limit=0.01)
    bulge.axis_ratio = af.UniformPrior(lower_limit=0.79, upper_limit=0.81)
    bulge.phi = af.UniformPrior(lower_limit=-1.0, upper_limit=1.0)
    bulge.intensity = af.UniformPrior(lower_limit=0.99, upper_limit=1.01)
    bulge.effective_radius = af.UniformPrior(lower_limit=1.25, upper_limit=1.35)

    phase1 = ag.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, bulge=ag.lp.EllipticalSersic)
        ),
        search=search,
    )

    phase1.search.const_efficiency_mode = True
    phase1.search.n_live_points = 40
    phase1.search.facc = 0.8

    class GridPhase(af.as_grid_search(phase_class=ag.PhaseImaging, parallel=False)):
        @property
        def grid_priors(self):
            return [self.model.galaxies.light.bulge.effective_radius]

    phase2 = GridPhase(
        phase_name="phase_2",
        folders=folders,
        galaxies=dict(galaxy=ag.GalaxyModel(redshift=0.5, bulge=bulge)),
        number_of_steps=2,
        search=search,
    )

    phase2.search.const_efficiency_mode = True
    phase2.search.n_live_points = 10
    phase2.search.facc = 0.5

    phase3 = ag.PhaseImaging(
        phase_name="phase_3",
        folders=folders,
        galaxies=dict(galaxy=phase2.result.model.galaxies.light),
        search=search,
    )

    phase3.search.const_efficiency_mode = True
    phase3.search.n_live_points = 40
    phase3.search.facc = 0.8

    return ag.PipelineDataset(name, phase1, phase2, phase3)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
