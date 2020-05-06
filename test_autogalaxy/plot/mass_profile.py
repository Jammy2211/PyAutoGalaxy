import autoarray as aa
import autogalaxy as ag
import autogalaxy.plot as aplt

grid = aa.Grid.uniform(shape_2d=(101, 101), pixel_scales=0.05)

mass_profile = ag.mp.EllipticalIsothermal(einstein_radius=1.0, axis_ratio=0.9)

aplt.MassProfile.convergence(mass_profile=mass_profile, grid=grid)
