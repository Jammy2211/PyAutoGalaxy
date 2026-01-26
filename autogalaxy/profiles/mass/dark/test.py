import mcr_util
import numpy as np

concentration, cosmic_average_density, critical_surface_density, kpc_per_arcsec = mcr_util._ludlow16_cosmology_callback(mass_at_200=1.0e9,redshift_object=0.6,
                 redshift_source=2.5)

print(concentration
      , cosmic_average_density
      , critical_surface_density
      , kpc_per_arcsec)

r_200 = (3 * 1.0e9 / (4 * 200 * cosmic_average_density * np.pi))**(1/3)
scale_radius = r_200 / concentration / kpc_per_arcsec

print('r_200', r_200)
print('scale_radius', scale_radius)