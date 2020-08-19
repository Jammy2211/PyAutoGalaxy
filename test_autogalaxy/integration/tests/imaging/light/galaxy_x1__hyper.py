import autofit as af
import autogalaxy as ag
from test_autogalaxy.integration.tests.imaging import runner

test_type = "bulge"
test_name = "galaxy_x1__hyper"
data_name = "galaxy_x1__dev_vaucouleurs"
instrument = "vro"


def make_pipeline(name, folders, search=af.DynestyStatic()):

    pipeline_name = "pipeline__hyper"

    setup.folders.append(pipeline_name)
    folders.append("setup")

    phase1 = ag.PhaseImaging(
        phase_name="phase_1",
        folders=folders,
        galaxies=dict(
            galaxy=ag.GalaxyModel(redshift=0.5, light=ag.lp.EllipticalSersic)
        ),
        settings=ag.SettingsPhaseImaging(grid_class=ag.Grid),
        search=search,
    )

    phase1 = phase1.extend_with_multiple_hyper_phases(
        hyper_galaxies_search=True,
        include_background_sky=True,
        include_background_noise=True,
    )

    return ag.PipelineDataset(name, phase1)


if __name__ == "__main__":
    import sys

    runner.run(sys.modules[__name__])
