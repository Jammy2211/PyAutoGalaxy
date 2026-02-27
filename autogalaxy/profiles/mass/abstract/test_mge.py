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

nfw = ag.mp.gNFW(
        centre=(0.0, 0.0),
        kappa_s=1.0,
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.3, angle=100.0),
        inner_slope=0.5,
        scale_radius=8.0,
    )

radii_min = nfw.scale_radius / 2000.0
radii_max = nfw.scale_radius * 30.0
log_sigmas = jnp.linspace(jnp.log(radii_min), jnp.log(radii_max), 20)
sigmas = jnp.exp(log_sigmas)

deflections_via_integral = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
    )

mge_decomp = mge(mass_profile=nfw)

deflections_via_mge_np = nfw.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
    )

deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np,
        sigma_log_list = sigmas,
        sigmas_factor = nfw.axis_ratio(np),
        three_D = True
    )

print('nfw via integral:', deflections_via_integral)
print('nfw via mge_np:', deflections_via_mge_np)
print('nfw via mge:', deflections_via_mge)

#assert np.allclose(deflections_via_integral, deflections_via_mge, 1.0e-3)

nfw = ag.mp.gNFW(
        centre=(0.3, 0.2),
        kappa_s=2.5,
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.5, angle=100.0),
        inner_slope=1.5,
        scale_radius=4.0,
    )

radii_min = nfw.scale_radius / 20000.0
radii_max = nfw.scale_radius * 30.0
log_sigmas = jnp.linspace(jnp.log(radii_min), jnp.log(radii_max), 20)
sigmas = jnp.exp(log_sigmas)

deflections_via_integral = nfw.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
    )
mge_decomp = mge(mass_profile=nfw)

deflections_via_mge_np = nfw.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]])
    )

deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np,
        sigma_log_list = sigmas,
        sigmas_factor = nfw.axis_ratio(np),
        three_D = True
    )

print('nfw via integral:', deflections_via_integral)
print('nfw via mge_np:', deflections_via_mge_np)
print('nfw via mge:', deflections_via_mge)

#assert np.allclose(deflections_via_integral, deflections_via_mge, 1.0e-3)


mp = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )

radii_min = mp.effective_radius / 100.0
radii_max = mp.effective_radius * 20.0
log_sigmas = jnp.linspace(jnp.log(radii_min), jnp.log(radii_max), 20)
sigmas = jnp.exp(log_sigmas)

deflections_via_integral = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
)
mge_decomp = mge(mass_profile=mp)

deflections_via_mge_np = mp.deflections_2d_via_mge_from(
         grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
     )

deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]]),
        xp = np,
        sigma_log_list = sigmas,
        sigmas_factor = np.sqrt(mp.axis_ratio(np)),
        three_D = False,
)

print('sersic via integral:', deflections_via_integral)
print('sersic via mge_np:', deflections_via_mge_np)
print('sersic via mge:', deflections_via_mge)

#assert np.allclose(deflections_via_integral, deflections_via_mge, 1.0e-3)

mp = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.17142, 0.385116),
        intensity=10.0,
        effective_radius=0.2,
        sersic_index=3.0,
        mass_to_light_ratio=1.0,
    )

deflections_via_integral = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

mge_decomp = mge(mass_profile=mp)

deflections_via_mge_np = mp.deflections_2d_via_mge_from(
         grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
     )

deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]]),
        xp = np,
        sigma_log_list = sigmas,
        sigmas_factor = np.sqrt(mp.axis_ratio(np)),
        three_D = False,
)

print('sersic via integral:', deflections_via_integral)
print('sersic via mge_np:', deflections_via_mge_np)
print('sersic via mge:', deflections_via_mge)

#assert np.allclose(deflections_via_integral, deflections_via_mge, 1.0e-3)

cnfw = ag.mp.cNFWSph(
    centre=(0.3, 0.2),
    kappa_s=0.05,
    scale_radius=1.1,
    core_radius=0.01
)

radii_min = cnfw.scale_radius / 1000.0
radii_max = cnfw.scale_radius * 200.0
log_sigmas = jnp.linspace(jnp.log(radii_min), jnp.log(radii_max), 20)
sigmas = jnp.exp(log_sigmas)

deflections_analytic = cnfw.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np)

mge_decomp = mge(mass_profile=cnfw)

deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np,
        sigma_log_list = sigmas,
        three_D = True
    )

print('cnfw via analytic:', deflections_analytic)
print('cnfw via mge:', deflections_via_mge)

#assert np.allclose(deflections_analytic, deflections_via_mge, 1.0e-3)

mp = ag.mp.Isothermal(
    centre=(-0.4, -0.2),
    ell_comps=(-0.27142, -0.385116),
    einstein_radius=0.2,
)

radii_min = mp.einstein_radius / 10000.0
radii_max = mp.einstein_radius * 500.0
log_sigmas = jnp.linspace(jnp.log(radii_min), jnp.log(radii_max), 20)
sigmas = jnp.exp(log_sigmas)

deflections_analytic = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np)

mge_decomp = mge(mass_profile=mp)

deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np,
        sigma_log_list = sigmas,
        sigmas_factor = mp.axis_ratio(np),
        three_D = False
    )

print('Isothermal via analytic:', deflections_analytic)
print('Isothermal via mge:', deflections_via_mge)

#assert np.allclose(deflections_analytic, deflections_via_mge, 1.0e-2)

mp = ag.mp.PowerLaw(
    centre=(-0.4, -0.2),
    ell_comps=(0.17142, -0.285116),
    einstein_radius=0.2,
    slope=1.8,
)

radii_min = mp.einstein_radius / 10000.0
radii_max = mp.einstein_radius * 1000.0
log_sigmas = jnp.linspace(jnp.log(radii_min), jnp.log(radii_max), 30)
sigmas = jnp.exp(log_sigmas)

deflections_analytic = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np)

mge_decomp = mge(mass_profile=mp)

deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1875, 0.1625]]),
        xp=np,
        sigma_log_list = sigmas,
        sigmas_factor = mp.axis_ratio(np),
        three_D = False
    )

print('PowerLaw via analytic:', deflections_analytic)
print('PowerLaw via mge:', deflections_via_mge)

#assert np.allclose(deflections_analytic, deflections_via_mge, 1.0e-3)

mp = ag.mp.Chameleon(
    centre=(-0.4, -0.2),
    ell_comps=(0.17142, -0.285116),
    intensity=2.8
)

radii_min = mp.core_radius_0 / 10000.0
radii_max = mp.core_radius_0 * 10000.0
log_sigmas = jnp.linspace(jnp.log(radii_min), jnp.log(radii_max), 30)
sigmas = jnp.exp(log_sigmas)

deflections_analytic = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[-0.1875, 0.1625]]),
        xp=np)

mge_decomp = mge(mass_profile=mp)

deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[-0.1875, 0.1625]]),
        xp=np,
        sigma_log_list = sigmas,
        sigmas_factor = mp.axis_ratio(np),
        three_D = False
    )

print('Chameleon via analytic:', deflections_analytic)
print('Chameleon via mge:', deflections_via_mge)

mp = ag.mp.Gaussian(
    centre=(-0.4, -0.2),
    ell_comps=(0.17142, -0.285116),
    intensity=1.5,
    sigma=1.2,
    mass_to_light_ratio=1.1
)

radii_min = mp.sigma / 50000.0
radii_max = mp.sigma * 10
log_sigmas = jnp.linspace(jnp.log(radii_min), jnp.log(radii_max), 30)
sigmas = jnp.exp(log_sigmas)

deflections_analytic = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[-0.1875, 0.1625]]),
        xp=np)

deflections_via_integral = mp.deflections_2d_via_integral_from(grid=ag.Grid2DIrregular([[-0.1875, 0.1625]]),
        xp=np)

mge_decomp = mge(mass_profile=mp)

deflections_via_mge = mge_decomp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[-0.1875, 0.1625]]),
        xp=np,
        sigma_log_list = sigmas,
        sigmas_factor = 1.0,
        three_D = False
    )

print('Gaussian via analytic:', deflections_analytic)
print('Gaussian via integral:', deflections_via_integral)
print('Gaussian via mge:', deflections_via_mge)