import os
from test import integration_util

from autoconf import conf
import autofit as af
import autogalaxy as ag
import numpy as np

test_type = "galaxy_fit"
test_name = "deflections"

test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"
config_path = test_path + "config"
conf.instance = conf.Config(config_path=config_path, output_path=output_path)


def galaxy_fit_phase():

    pixel_scales = 0.1
    image_shape = (150, 150)

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    grid = ag.Grid.uniform(shape_2d=image_shape, pixel_scales=pixel_scales, sub_size=4)

    galaxy = ag.Galaxy(
        redshift=0.5,
        mass=ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )

    deflections = galaxy.deflections_from_grid(galaxies=[galaxy], grid=grid)

    noise_map = ag.Array.manual_2d
        sub_array_1d=np.ones(deflections[:, 0].shape), pixel_scales=pixel_scales
    )

    data_y = ag.GalaxyData(
        image=deflections[:, 0], noise_map=noise_map, pixel_scales=pixel_scales
    )
    data_x = ag.GalaxyData(
        image=deflections[:, 1], noise_map=noise_map, pixel_scales=pixel_scales
    )

    phase1 = ag.PhaseGalaxy(
        phase_name=test_name + "/",
        folders=folders,
        galaxies=dict(
            gal=ag.GalaxyModel(redshift=0.5, light=ag.mp.SphericalIsothermal)
        ),
        use_deflections=True,
        sub_size=4,
        search=af.DynestyStatic(),
    )

    phase1.run(galaxy_data=[data_y, data_x])


if __name__ == "__main__":
    galaxy_fit_phase()
