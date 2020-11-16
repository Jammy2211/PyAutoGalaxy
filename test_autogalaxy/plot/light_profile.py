import autogalaxy as ag
import autogalaxy as ag
import autogalaxy.plot as aplt

grid = ag.Grid.uniform(shape_2d=(101, 101), pixel_scales=0.05)

light_profile = ag.lp.EllipticalSersic(effective_radius=2.0, sersic_index=2.0)

aplt.LightProfile.image(light_profile=light_profile, grid=grid)
