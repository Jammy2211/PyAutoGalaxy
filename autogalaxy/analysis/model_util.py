import numpy as np
from typing import Optional, Tuple

import autofit as af


def mge_model_from(
    mask_radius: float,
    total_gaussians: int = 30,
    gaussian_per_basis: int = 1,
    centre_prior_is_uniform: bool = True,
    centre: Tuple[float, float] = (0.0, 0.0),
    centre_fixed: Optional[Tuple[float, float]] = None,
    centre_per_basis: bool = False,
    centre_sigma: float = 0.3,
    ell_comps_prior_is_uniform: bool = False,
    ell_comps_uniform_width: float = 0.2,
    ell_comps_sigma : float = 0.3,
    ell_comps_limit: float = 1.0,
    use_spherical: bool = False,
    sigma_min: float = 1e-4,
    order_bases: bool = False,
) -> af.Collection:
    """
    Construct a Multi-Gaussian Expansion (MGE) for the lens or source galaxy light.

    This model is designed as a "start here" configuration for lens modeling:

    - The lens and source light are represented by a Basis object composed of many
      Gaussian light profiles with fixed logarithmically spaced widths (`sigma`).
    - All Gaussians within each basis share common centres and ellipticity
      components, reducing degeneracy while retaining flexibility.
    - Users can combine with a lens mass model of their choice.

    When ``gaussian_per_basis > 1``, each basis receives **independent** ellipticity
    components (``ell_comps``), allowing the model to represent twisting or varying
    isophotes across different radial scales. Centres are **shared** across bases by
    default (the common case: one luminosity centre, complex isophotal shape). Set
    ``centre_per_basis=True`` to give each basis its own centre priors.

    Expected free-parameter counts (elliptical, ``use_spherical=False``):

    - ``gaussian_per_basis=1`` : 2 centre + 2 ell_comps = 4
    - ``gaussian_per_basis=K`` (shared centre) : 2 + 2K
    - ``gaussian_per_basis=K, centre_per_basis=True`` : 2K + 2K = 4K

    Spherical (``use_spherical=True``): no ell_comps, only centres.

    - Shared centre: 2.  Per-basis centre: 2K.

    Parameters
    ----------
    mask_radius
        The outer radius (in arcseconds) of the circular mask applied to the data.
        This determines the maximum Gaussian width (`sigma`) used in the MGE.
    total_gaussians
        Total number of Gaussian light profiles used in each basis.
    gaussian_per_basis
        Number of separate Gaussian bases. Each basis has ``total_gaussians``
        components sharing the same centre and ellipticity. Multiple bases allow
        independent ellipticity (and optionally centre) per radial scale group.
    centre_prior_is_uniform
        If True (default), centre priors are ``UniformPrior(±0.1)`` around
        ``centre``. If False, ``GaussianPrior`` with ``centre_sigma``.
    centre
        (y, x) centre in arcseconds used as the mean/midpoint for centre priors.
    centre_fixed
        If not None, fix all Gaussian centres to this (y, x) value instead of
        making them free parameters. Overrides ``centre_per_basis``.
    centre_per_basis
        If True, each basis gets independently drawn centre priors. If False
        (default), all bases share the same centre. Ignored when ``centre_fixed``
        is set.
    centre_sigma
        Sigma for ``GaussianPrior`` centre priors (used when
        ``centre_prior_is_uniform=False``).
    ell_comps_prior_is_uniform
        If True, ell_comps priors are ``UniformPrior``. If False (default),
        ``TruncatedGaussianPrior``.
    ell_comps_uniform_width
        Half-width for uniform ell_comps priors.
    ell_comps_sigma
        Sigma for truncated-Gaussian ell_comps priors.
    ell_comps_limit
        Half-width of the box the truncated-Gaussian ell_comps priors are truncated to,
        giving ``lower_limit=-ell_comps_limit`` and ``upper_limit=+ell_comps_limit``.
        Must satisfy ``0.0 < ell_comps_limit <= 1.0``; the default ``1.0`` is the full
        physical range. Uniform ell_comps priors are unaffected -- their width is set by
        ``ell_comps_uniform_width``.

        Callers tighten this box when the science case bounds the isophotal ellipticity:
        the Euclid strong-lens pipeline uses ``0.5`` for the lens light and ``0.7`` for
        the source. Setting the box *here* rather than overwriting the priors on the
        returned model keeps the prior objects the ones this function built, which
        matters because ``order_bases`` attaches an assertion that references them.
    use_spherical
        If True, use ``GaussianSph`` (no ell_comps). If False (default), use
        ``Gaussian`` with ellipticity.
    sigma_min
        The smallest Gaussian width (`sigma`) in arcseconds, which sets the lower end
        of the log-spaced sigma values. Defaults to ``1e-4``. Increase it (e.g. to a
        tenth of the pixel scale) to stop the basis wasting components on scales the
        data cannot resolve.
    order_bases
        If True, require the bases' shared ``ell_comps_1`` values to be strictly
        decreasing, ``basis_0 > basis_1 > ... > basis_{K-1}``, via ``K - 1`` assertions
        added to the returned model. Defaults to False (off until validated on
        production runs). A no-op when ``gaussian_per_basis == 1``; a ``ValueError``
        when ``use_spherical=True``, which has no ``ell_comps`` to order.

        **The symmetry.** With ``gaussian_per_basis=K > 1`` every basis holds the same
        ``total_gaussians`` linear Gaussians on the same fixed ``log10_sigma_list``, and
        differs from its siblings only in the ellipticity pair it carries. The bases are
        therefore *exchangeable*: permuting which basis holds which ``ell_comps`` leaves
        the likelihood exactly unchanged, so the posterior has ``K!`` identical modes and
        an unseeded search lands in whichever one it reaches first. Repeat fits of the
        same data then report the same physical solution under swapped labels, which
        breaks any downstream comparison that reads ``basis_0`` as a fixed component.
        Ordering the bases picks one labelling and deletes the other ``K! - 1`` copies;
        it removes no physical solution.

        **Why ``ell_comps_1`` and not the magnitude.** The key must separate the modes
        the search actually finds. On the Euclid phase-4 tiles the two bases commonly sit
        against opposite edges of the ell_comps box, e.g. ``(0.007, -0.500)`` and
        ``(-0.023, 0.497)``: equal in magnitude to within 0.003, but separated by ~1.0 in
        ``ell_comps_1``. A magnitude key would cut straight through that pair and forbid
        the very solution the data prefer, whereas the ``cos 2phi`` component separates it
        cleanly.

        **Blind band.** No continuous key is exact for every configuration -- another tile
        has the two bases only ``0.01`` apart in ``ell_comps_1``, inside the width the
        search resolves. A result whose bases differ by a small ``|delta ell_comps_1|`` is
        one where the ordering did not resolve the symmetry; read it as undetermined
        labelling rather than as an ordered answer.

        **Consequences of an assertion being part of the model.** It enters the PyAutoFit
        identifier, so turning ``order_bases`` on gives an otherwise identical fit a new
        ``unique_id`` and a fresh output directory. It also makes the model an invalid
        *target* for ``take_attributes``: PyAutoFit's ``assert_no_assertions`` refuses to
        copy attributes into a model that already carries assertions, so build the ordered
        model after any such prior-passing step. Enforcement is backend-specific but has
        the same outcome -- NumPy raises ``af.exc.FitException`` from ``check_assertions``
        and the search resamples; JAX cannot raise inside a trace and instead evaluates
        the assertions as a traced boolean and maps a violating model to the resample
        figure of merit (PyAutoFit #1583).

    Returns
    -------
    af.Model
        An ``autofit.Model`` wrapping a ``Basis`` of linear Gaussians.
    """

    import os

    if os.environ.get("PYAUTO_SMALL_DATASETS") == "1":
        total_gaussians = 2
        # A single basis leaves nothing to order, so `order_bases` degrades to a no-op
        # under the small-dataset shortcut rather than changing what it asserts.
        gaussian_per_basis = 1

    from autogalaxy.profiles.light.linear import Gaussian, GaussianSph
    from autogalaxy.profiles.basis import Basis

    if not 0.0 < ell_comps_limit <= 1.0:
        raise ValueError(
            f"mge_model_from requires 0.0 < ell_comps_limit <= 1.0, got "
            f"{ell_comps_limit}."
        )

    if order_bases and use_spherical:
        raise ValueError(
            "mge_model_from cannot order bases when use_spherical=True, because "
            "spherical Gaussians have no ell_comps to order."
        )

    if sigma_min <= 0.0:
        raise ValueError(
            f"mge_model_from requires sigma_min > 0.0, got {sigma_min}."
        )

    if sigma_min > mask_radius:
        raise ValueError(
            f"mge_model_from requires sigma_min <= mask_radius, got sigma_min="
            f"{sigma_min} and mask_radius={mask_radius}."
        )

    # The sigma values of the Gaussians are fixed to log-spaced values spanning
    # `sigma_min` (default 0.0001") to the mask radius.
    log10_sigma_list = np.linspace(
        np.log10(sigma_min), np.log10(mask_radius), total_gaussians
    )

    if use_spherical:
        model_cls = GaussianSph
    else:
        model_cls = Gaussian

    def _make_centre_priors():
        if centre_fixed is not None:
            return centre_fixed[0], centre_fixed[1]
        elif centre_prior_is_uniform:
            return (
                af.UniformPrior(
                    lower_limit=centre[0] - 0.1, upper_limit=centre[0] + 0.1
                ),
                af.UniformPrior(
                    lower_limit=centre[1] - 0.1, upper_limit=centre[1] + 0.1
                ),
            )
        else:
            return (
                af.GaussianPrior(mean=centre[0], sigma=centre_sigma),
                af.GaussianPrior(mean=centre[1], sigma=centre_sigma),
            )

    def _make_ell_comps_priors():
        if ell_comps_prior_is_uniform:
            return (
                af.UniformPrior(lower_limit=-ell_comps_uniform_width, upper_limit=ell_comps_uniform_width),
                af.UniformPrior(lower_limit=-ell_comps_uniform_width, upper_limit=ell_comps_uniform_width),
            )
        else:
            return (
                af.TruncatedGaussianPrior(mean=0.0, sigma=ell_comps_sigma, lower_limit=-ell_comps_limit, upper_limit=ell_comps_limit),
                af.TruncatedGaussianPrior(mean=0.0, sigma=ell_comps_sigma, lower_limit=-ell_comps_limit, upper_limit=ell_comps_limit),
            )

    # Shared centre priors (used when centre_per_basis=False).
    if not centre_per_basis or centre_fixed is not None:
        shared_centre_0, shared_centre_1 = _make_centre_priors()

    bulge_gaussian_list = []

    # The shared `ell_comps_1` prior of each basis, in basis order, used as the ordering
    # key when `order_bases` is True.
    ell_comps_1_list = []

    for j in range(gaussian_per_basis):

        # Per-basis centre priors when requested.
        if centre_per_basis and centre_fixed is None:
            centre_0, centre_1 = _make_centre_priors()
        else:
            centre_0, centre_1 = shared_centre_0, shared_centre_1

        # Per-basis ell_comps priors (always independent across bases).
        if not use_spherical:
            ell_comps_0, ell_comps_1 = _make_ell_comps_priors()
            ell_comps_1_list.append(ell_comps_1)

        gaussian_list = af.Collection(
            af.Model(model_cls) for _ in range(total_gaussians)
        )

        for i, gaussian in enumerate(gaussian_list):
            gaussian.centre.centre_0 = centre_0
            gaussian.centre.centre_1 = centre_1
            if not use_spherical:
                gaussian.ell_comps.ell_comps_0 = ell_comps_0
                gaussian.ell_comps.ell_comps_1 = ell_comps_1
            gaussian.sigma = 10 ** log10_sigma_list[i]

        bulge_gaussian_list += gaussian_list

    # The Basis object groups many light profiles together into a single model component.

    model = af.Model(
        Basis,
        profile_list=bulge_gaussian_list,
    )

    # Break the label symmetry between exchangeable bases by requiring their shared
    # `ell_comps_1` values to be strictly decreasing. Attaching the assertions to the
    # returned `Basis` model (rather than to a parent) means they travel with the
    # component, and PyAutoFit's `gathered_assertions` finds them wherever the component
    # is placed in a larger model.
    if order_bases:
        for j in range(len(ell_comps_1_list) - 1):
            model.add_assertion(
                ell_comps_1_list[j] > ell_comps_1_list[j + 1],
                name=f"mge_basis_{j}_ell_comps_1_gt_basis_{j + 1}",
            )

    return model


def mge_point_model_from(
    pixel_scales: float,
    total_gaussians: int = 10,
    centre: Tuple[float, float] = (0.0, 0.0),
    sigma_min: float = 0.01,
) -> af.Model:
    """
    Construct a Multi-Gaussian Expansion (MGE) model for a compact or unresolved
    point-like component (e.g. a nuclear starburst, AGN, or unresolved bulge).

    The model is composed of ``total_gaussians`` linear Gaussians whose sigma values
    are logarithmically spaced between ``sigma_min`` and twice the pixel scale.
    All Gaussians share the same centre and ellipticity components, keeping the
    parameter count low while capturing a realistic PSF-convolved point source.

    Parameters
    ----------
    pixel_scales
        The pixel scale of the image in arcseconds per pixel.  The maximum Gaussian
        width is set to ``2 * pixel_scales`` so that the model is compact relative to
        the resolution of the data.
    total_gaussians
        Number of Gaussian components in the basis.
    centre
        (y, x) centre of the point source in arc-seconds.  A ±0.1 arcsecond uniform
        prior is placed on each coordinate.
    sigma_min
        The smallest Gaussian width (`sigma`) in arcseconds, which sets the lower end
        of the log-spaced sigma values. Defaults to ``0.01``. Increase it (e.g. to a
        tenth of the pixel scale) to stop the basis wasting components on scales the
        data cannot resolve.

    Returns
    -------
    af.Model
        An ``autofit.Model`` wrapping a ``Basis`` of linear Gaussians.
    """

    from autogalaxy.profiles.light.linear import Gaussian
    from autogalaxy.profiles.basis import Basis

    if total_gaussians < 1:
        raise ValueError(
            f"mge_point_model_from requires total_gaussians >= 1, got {total_gaussians}."
        )

    if pixel_scales <= 0:
        raise ValueError(
            f"mge_point_model_from requires pixel_scales > 0, got {pixel_scales}."
        )

    if sigma_min <= 0.0:
        raise ValueError(
            f"mge_point_model_from requires sigma_min > 0.0, got {sigma_min}."
        )

    # Sigma values are logarithmically spaced between `sigma_min` (default 0.01")
    # and twice the pixel scale, with a floor to keep the upper end of the list
    # at or above `sigma_min` when the pixel scale is very small.
    max_sigma = max(2.0 * pixel_scales, sigma_min)

    log10_sigma_list = np.linspace(
        np.log10(sigma_min), np.log10(max_sigma), total_gaussians
    )
    centre_0 = af.UniformPrior(lower_limit=centre[0] - 0.1, upper_limit=centre[0] + 0.1)
    centre_1 = af.UniformPrior(lower_limit=centre[1] - 0.1, upper_limit=centre[1] + 0.1)

    gaussian_list = af.Collection(af.Model(Gaussian) for _ in range(total_gaussians))

    for i, gaussian in enumerate(gaussian_list):
        gaussian.centre.centre_0 = centre_0
        gaussian.centre.centre_1 = centre_1
        gaussian.ell_comps = gaussian_list[0].ell_comps
        gaussian.sigma = 10 ** log10_sigma_list[i]

    return af.Model(Basis, profile_list=gaussian_list)


def hilbert_pixels_from_pixel_scale(pixel_scale: float) -> int:
    """
    Return the number of Hilbert-curve pixels appropriate for a given image pixel scale.

    The Hilbert pixel count controls the resolution of the Hilbert-curve ordering used
    in adaptive source-plane pixelizations. Finer pixel scales resolve smaller angular
    features and therefore benefit from a higher Hilbert resolution.

    Parameters
    ----------
    pixel_scale
        The pixel scale of the image in arcseconds per pixel.

    Returns
    -------
    int
        The recommended number of Hilbert pixels.
    """
    if not np.isfinite(pixel_scale) or pixel_scale <= 0:
        raise ValueError(
            f"hilbert_pixels_from_pixel_scale requires pixel_scale to be finite and > 0, got {pixel_scale}."
        )

    if pixel_scale > 0.06:
        return 1000
    elif pixel_scale > 0.04:
        return 1250
    elif pixel_scale >= 0.03:
        return 1500
    else:
        return 1750


SIMULATOR_RANDOM_GALAXY_SUMMARY = (
    "Each simulated galaxy draws a fresh bulge from: "
    "signal-to-noise ratio in [20, 60], "
    "effective radius in [1.0, 5.0] arcsec, "
    "sersic index in [3.5, 4.5], "
    "ell_comps each ~ Normal(0, 0.2) clipped to [-1, 1]."
)


def random_galaxy_for_simulation_from(
    rng: Optional[np.random.Generator] = None,
) -> "Galaxy":
    """
    Sample a random ``Galaxy`` instance with an SNR-normalised Sersic bulge.

    This helper is for **synthetic data generation only** — it draws each
    parameter directly from a numpy ``Generator`` and returns a concrete
    ``Galaxy`` instance whose bulge is an ``lp_snr.Sersic`` with a target
    signal-to-noise ratio. The SNR profile internally back-computes the
    matching ``intensity`` from the simulator's noise level.

    Do **not** use the returned ``Galaxy`` as a fitting model — SNR is a
    property of the data, not a parameter you fit. For fitting, build a
    regular ``af.Model(ag.lp.Sersic)`` with an explicit ``intensity`` prior.

    Parameters
    ----------
    rng
        Optional ``numpy.random.Generator`` for reproducible sampling. If
        ``None`` (default) a fresh ``default_rng()`` is created on each call,
        so each call returns a different galaxy.

    Returns
    -------
    Galaxy
        A ``Galaxy`` at redshift 0.5 with a single ``lp_snr.Sersic`` bulge.
    """
    from autogalaxy.profiles.light.snr import Sersic
    from autogalaxy.galaxy.galaxy import Galaxy

    rng = rng if rng is not None else np.random.default_rng()

    def _clipped_ell_comp() -> float:
        return float(np.clip(rng.normal(0.0, 0.2), -1.0, 1.0))

    bulge = Sersic(
        centre=(0.0, 0.0),
        ell_comps=(_clipped_ell_comp(), _clipped_ell_comp()),
        effective_radius=float(rng.uniform(1.0, 5.0)),
        sersic_index=float(rng.uniform(3.5, 4.5)),
        signal_to_noise_ratio=float(rng.uniform(20.0, 60.0)),
    )

    return Galaxy(redshift=0.5, bulge=bulge)
