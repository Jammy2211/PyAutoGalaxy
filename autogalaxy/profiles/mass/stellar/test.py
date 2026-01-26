import jax.numpy as jnp

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])

mp = ag.mp.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.05263),
    intensity=1.0,
    sigma=3.0,
    mass_to_light_ratio=1.0,
)

deflections = mp.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[1.0, 0.0]]),
    xp=jnp
)

print(deflections[0, 0])
print(deflections[0, 1])

mp = ag.mp.Gaussian(
    centre=(0.0, 0.0),
    ell_comps=(0.0, 0.111111),
    intensity=1.0,
    sigma=5.0,
    mass_to_light_ratio=1.0,
)

deflections = mp.deflections_2d_via_analytic_from(
    grid=ag.Grid2DIrregular([[0.5, 0.2]])
)

print(deflections[0, 0])
print(deflections[0, 1])