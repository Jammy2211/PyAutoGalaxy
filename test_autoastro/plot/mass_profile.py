import autoarray as aa
import autoastro as aast
import autoastro.plot as aplt

grid = aa.grid.uniform(shape_2d=(101, 101), pixel_scales=0.05)

mass_profile = aast.mp.EllipticalIsothermal(einstein_radius=1.0, axis_ratio=0.9)

aplt.mp.convergence(mass_profile=mass_profile, grid=grid)
