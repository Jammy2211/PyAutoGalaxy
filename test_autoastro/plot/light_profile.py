import autoarray as aa
import autoastro as aast
import autoastro.plot as aplt

grid = aa.grid.uniform(shape_2d=(101, 101), pixel_scales=0.05)

light_profile = aast.lp.EllipticalSersic(effective_radius=2.0, sersic_index=2.0)

aplt.lp.profile_image(light_profile=light_profile, grid=grid)
