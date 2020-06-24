import os
from test import integration_util

import autofit as af
import autogalaxy as ag
import numpy as np

test_type = "galaxy_fit"
test_name = "convergence"

test_path = "{}/../../".format(os.path.dirname(os.path.realpath(__file__)))
output_path = test_path + "output/"


def galaxy_fit_phase():

    pixel_scales = 0.1
    image_shape = (150, 150)

    integration_util.reset_paths(test_name=test_name, output_path=output_path)

    grid = ag.Grid.uniform(shape_2d=image_shape, pixel_scales=pixel_scales, sub_size=4)

    galaxy = ag.Galaxy(
        redshift=0.5,
        mass=ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0),
    )

    convergence = galaxy.convergence_from_grid(galaxies=[galaxy], grid=grid)

    noise_map = ag.Array.manual_2d
        sub_array_1d=np.ones(convergence.shape), pixel_scales=pixel_scales
    )

    data = ag.GalaxyData(
        image=convergence, noise_map=noise_map, pixel_scales=pixel_scales
    )

    phase1 = ag.PhaseGalaxy(
        phase_name=test_name + "/",
        folders=folders,
        galaxies=dict(
            gal=ag.GalaxyModel(redshift=0.5, light=ag.mp.SphericalIsothermal)
        ),
        use_convergence=True,
        sub_size=4,
        search=af.DynestyStatic(),
    )

    phase1.run(galaxy_data=[data])


if __name__ == "__main__":
    galaxy_fit_phase()
