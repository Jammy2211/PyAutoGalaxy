from mge import MGEDecomposer as mge
from mge_numpy import MassProfileMGE as mge_np
from autogalaxy.profiles.mass.stellar.gaussian import Gaussian
import numpy as np
import jax.numpy as jnp
import jax
import autogalaxy as ag
import autolens as al
import autoarray as aa

grid = al.Grid2D.uniform(shape_native=(11, 11), pixel_scales=0.2,
                             over_sample_size=8)

jax.config.update("jax_enable_x64", True)


kesi_np = mge.kesi(100, xp=np)
kesi_jnp = mge.kesi(100, xp=jnp)
assert np.allclose(kesi_np, kesi_jnp)

eta_np_np = mge_np.eta(10)
eta_np = mge.eta(10, xp=np)
eta_jnp = mge.eta(10, xp=jnp)
assert np.allclose(eta_np_np, eta_jnp)
assert np.allclose(eta_np, eta_jnp)

mp = ag.mp.gNFW(
        centre=(0.0, 0.0), ell_comps=(0.1,-0.2), kappa_s=1.0, inner_slope=0.5, scale_radius=8.0
    )

def f(x):
    return x**(-2)

def f_ell(y, x):
    return jnp.sqrt(y**2 + x**2)**(-2)

sigmas = [0.01, 0.1, 1.0, 10.0]

mge_decomp = mge(mass_profile=mp)

print(mge_decomp.deflections_2d_via_mge_from(grid=grid, sigma_log_list=sigmas, xp=jnp))


# amps, sigmas = obj2.decompose_convergence_ell_via_mge(xp=jnp)
# print(amps)
# deflection_angles = (amps[:, None]* sigmas[:, None] * jnp.sqrt((2.0 * jnp.pi) / (1.0 - obj.axis_ratio(jnp)**2.0))
#                 * obj.zeta_from(grid=grid, xp=jnp)
#         )
#
# deflections =jnp.sum(deflection_angles, axis=0)
#
# deflections = jnp.multiply(
#                 1.0, jnp.vstack((-1.0 * jnp.imag(deflections), jnp.real(deflections))).T
 #           )
# print(deflections)
# deflections = obj.rotated_grid_from_reference_frame_from(deflections, xp=jnp)
# print(deflections)

#print(jnp.vstack((-1.0 * jnp.imag(deflection), jnp.real(deflection))).T)

# profile_func = lambda y, x: obj2.deflections_2d_gaussian_via_mge(
#     y,
#     x,
#     intensity=0.1,

#     sigma=1.0,
#     mass_to_light_ratio=2.0,
# )

#print(obj2.deflections_2d_via_mge_from(grid=grid, profile_func=obj2.deflections_2d_gaussian_via_mge, sigma_log_list=sigmas, xp=jnp))


#print(obj2.deflection_2d_gaussian_via_mge(grid=grid, xp=jnp, sigma_log_list=sigmas))

# obj3 = MassProfileMGE(centre=(0.0, 0.0), ell_comps=(0.1, -0.2))
#
# print(obj3.deflection_2d_gaussian_via_mge(grid=grid, xp=jnp, sigma_log_list=sigmas))



