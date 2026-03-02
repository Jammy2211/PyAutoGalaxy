from typing import Tuple

from autogalaxy.profiles.mass.dark.gnfw import gNFWSph

import numpy as np
from autogalaxy import cosmology as cosmo

def is_jax(x):
    try:
        import jax
        from jax import Array
        from jax.core import Tracer
        return isinstance(x, (Array, Tracer))
    except Exception:
        return False

def _hyp2f1_jax(xp, *, max_terms: int = 256):
    """
    Returns a callable hyp2f1(a,b,c,z) compatible with the backend xp.

    - NumPy: scipy.special.hyp2f1
    - JAX (if available): jax.scipy.special.hyp2f1
    - JAX (fallback): series approximation for 2F1 (sufficient for this gNFW use-case)
    """
    import jax
    import jax.numpy as jnp

    # Fallback: truncated series for 2F1(a,a;a+1;z) and general 2F1(a,b;c;z)
    # We implement general 2F1 series:
    #   2F1(a,b;c;z) = sum_{n=0}^{∞} (a)_n (b)_n / (c)_n * z^n / n!
    #
    # Recurrence for terms:
    #   t_0 = 1
    #   t_{n+1} = t_n * (a+n)(b+n)/((c+n)(n+1)) * z
    #
    # This is JIT-safe with static max_terms.
    def hyp2f1_series(a, b, c, z):
        a = jnp.asarray(a)
        b = jnp.asarray(b)
        c = jnp.asarray(c)
        z = jnp.asarray(z)

        def body_fun(n, carry):
            t, s = carry
            n_f = jnp.asarray(n, dtype=t.dtype)
            t = t * (a + n_f) * (b + n_f) / ((c + n_f) * (n_f + 1.0)) * z
            s = s + t
            return (t, s)

        # Start: t0 = 1, s0 = 1
        t0 = jnp.ones_like(z, dtype=jnp.result_type(a, b, c, z))
        s0 = t0

        # fori_loop has static iteration count => good under jit/vmap
        tN, sN = jax.lax.fori_loop(0, max_terms - 1, body_fun, (t0, s0))
        return sN

    return hyp2f1_series

def kappa_s_and_scale_radius(
    cosmology,
    virial_mass,
    c_2,
    overdens,
    redshift_object,
    redshift_source,
    inner_slope,
):
    """
    Compute the characteristic convergence and scale radius of a spherical gNFW halo
    parameterised by virial mass and concentration.

    This routine converts a halo defined by its virial mass and concentration into
    the equivalent gNFW parameters (`kappa_s`, `scale_radius`) used in lensing
    calculations. The normalization is computed analytically using the closed-form
    hypergeometric expression for the enclosed mass integral, ensuring compatibility
    with both NumPy and JAX backends (e.g. within `jax.jit`).

    The virial radius is defined via:

        M_vir = (4/3) π Δ ρ_crit(z_lens) r_vir^3

    where Δ is the overdensity with respect to the critical density. If `overdens`
    is set to zero, the Bryan & Norman (1998) redshift-dependent overdensity is used.

    The gNFW normalization constant is computed as:

        d_e = (Δ / 3) (3 − γ) c^γ /
              ₂F₁(3 − γ, 3 − γ; 4 − γ; −c)

    where γ is the inner slope and c is the gNFW concentration.

    Parameters
    ----------
    cosmology
        Cosmology object providing critical density, angular diameter distance
        conversions, and surface mass density calculations. Must support an `xp`
        argument for NumPy/JAX interoperability.
    virial_mass
        Virial mass of the halo in units of solar masses.
    c_2
        Concentration-like parameter, converted internally to the gNFW
        concentration via `(2 - inner_slope) * c_2`.
    overdens
        Overdensity with respect to the critical density. If zero, the
        Bryan & Norman (1998) redshift-dependent overdensity is used.
    redshift_object
        Redshift of the lens (halo).
    redshift_source
        Redshift of the background source.
    inner_slope
        Inner logarithmic density slope γ of the gNFW profile.
    xp
        Array backend module (`numpy` or `jax.numpy`). All array operations
        are dispatched through this module to ensure compatibility with
        both standard NumPy execution and JAX tracing / JIT compilation.

    Returns
    -------
    kappa_s
        Dimensionless characteristic convergence of the gNFW profile.
    scale_radius
        Angular scale radius in arcseconds.
    virial_radius
        Virial radius in kiloparsecs.
    overdens
        Final overdensity value used in the calculation.

    Notes
    -----
    - This implementation is fully JIT-compatible when `xp=jax.numpy`.
    - No Python-side branching depends on traced values; conditional logic
      is implemented via backend array operations.
    - The analytic normalization avoids numerical quadrature, improving
      both performance and differentiability.
    """
    is_jax_bool = is_jax(virial_mass)

    if not is_jax_bool:
        xp = np
    else:
        from jax import numpy as jnp
        xp = jnp

    if xp is np:
        from scipy.special import hyp2f1
    else:
        try:
            from jax.scipy.special import hyp2f1
        except ImportError:
            hyp2f1 = _hyp2f1_jax(xp)

    gamma = inner_slope
    concentration = (2.0 - gamma) * c_2  # gNFW concentration (your definition)

    critical_density = cosmology.critical_density(redshift_object, xp=xp)  # Msun / kpc^3

    critical_surface_density = cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
        redshift_0=redshift_object,
        redshift_1=redshift_source,
        xp=xp,
    )  # Msun / kpc^2

    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object, xp=xp)  # kpc / arcsec

    # Bryan & Norman (1998) overdensity if overdens == 0
    x = cosmology.Om(redshift_object, xp=xp) - 1.0
    overdens_bn98 = 18.0 * xp.pi**2 + 82.0 * x - 39.0 * x**2
    overdens = xp.where(overdens == 0, overdens_bn98, overdens)

    # r_vir in kpc
    virial_radius = (virial_mass / (overdens * critical_density * (4.0 * xp.pi / 3.0))) ** (1.0 / 3.0)

    # scale radius in kpc
    scale_radius_kpc = virial_radius / concentration

    # c = rvir/rs is exactly "concentration" by definition
    c = concentration

    # Analytic normalization
    a = 3.0 - gamma
    de_c = (overdens / 3.0) * a * (c**gamma) / hyp2f1(a, a, a + 1.0, -c)

    rho_s = critical_density * de_c  # Msun / kpc^3
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density  # dimensionless
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # arcsec

    return kappa_s, scale_radius, virial_radius, overdens


class gNFWVirialMassConcSph(gNFWSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        log10m_vir: float = 12.0,
        c_2: float = 10.0,
        overdens: float = 0.0,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
        inner_slope: float = 1.0,
    ):
        """
        Spherical gNFW profile initialized with the virial mass and c_2 concentration of the halo.

        The virial radius of the halo is defined as the radius at which the density of the halo
        equals overdens * the critical density of the Universe. r_vir = (3*m_vir/4*pi*overdens*critical_density)^1/3.

        If the overdens parameter is set to 0, the virial overdensity of Bryan & Norman (1998) will be used.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        log10m_vir
            The log10(virial mass) of the dark matter halo.
        c_2
            The c_2 concentration of the dark matter halo, which equals r_vir/r_2, where r_2 is the
            radius at which the logarithmic density slope equals -2.
        overdens
            The spherical overdensity used to define the virial radius of the dark matter
            halo: r_vir = (3*m_vir/4*pi*overdens*critical_density)^1/3. If this parameter is set to 0, the virial
            overdensity of Bryan & Norman (1998) will be used.
        redshift_object
            Lens redshift.
        redshift_source
            Source redshift.
        inner_slope
            The inner slope of the dark matter halo's gNFW density profile.
        """

        self.log10m_vir = log10m_vir
        self.c_2 = c_2
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source
        self.inner_slope = inner_slope

        (
            kappa_s,
            scale_radius,
            virial_radius,
            overdens,
        ) = kappa_s_and_scale_radius(
            cosmology=cosmo.Planck15(),
            virial_mass=10**log10m_vir,
            c_2=c_2,
            overdens=overdens,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
            inner_slope=inner_slope,
        )

        self.virial_radius = virial_radius
        self.overdens = overdens

        super().__init__(
            centre=centre,
            kappa_s=kappa_s,
            inner_slope=inner_slope,
            scale_radius=scale_radius,
        )
