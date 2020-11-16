import autogalaxy as ag
import autogalaxy as ag
import autogalaxy.plot as aplt

grid = ag.Grid.uniform(shape_2d=(101, 101), pixel_scales=0.05)

mass_profile = ag.mp.EllipticalIsothermal(
    einstein_radius=1.0, elliptical_comps=(0.0, 0.05263)
)

aplt.MassProfile.convergence(mass_profile=mass_profile, grid=grid)
