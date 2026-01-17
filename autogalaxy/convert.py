import numpy as np
from typing import Tuple


def ell_comps_from(axis_ratio: float, angle: float, xp=np) -> Tuple[float, float]:
    """
    Returns the elliptical components e1 and e2 of a light or mass profile from an input angle in degrees and axis
    ratio.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-axis_ratio)/(1+axis_ratio) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-axis_ratio)/(1+axis_ratio) * cos(2 * angle)

    Which are the values this function returns.

    Parameters
    ----------
    axis_ratio
        Ratio of light profiles ellipse's minor and major axes (b/a).
    angle
        Rotation angle of light profile counter-clockwise from positive x-axis.
    """
    angle *= xp.pi / 180.0
    fac = (1 - axis_ratio) / (1 + axis_ratio)
    ellip_y = fac * xp.sin(2 * angle)
    ellip_x = fac * xp.cos(2 * angle)
    return (ellip_y, ellip_x)


def axis_ratio_and_angle_from(
    ell_comps: Tuple[float, float], xp=np
) -> Tuple[float, float]:
    """
    Returns the axis-ratio and position angle in degrees (-45 < angle < 135.0) from input elliptical components e1
    and e2 of a light or mass profile.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-axis_ratio)/(1+axis_ratio) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-axis_ratio)/(1+axis_ratio) * cos(2 * angle)

    The axis-ratio and angle are therefore given by:

    axis_ratio = (1 - fac) / (1 + fac)
    angle = 0.5 * arctan(ell_comps[0] / ell_comps[1])

    where `fac = sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2).

    This function returns the axis-ratio and angle in degrees.

    An additional check is performed which requires the angle is between -45 and 135 degrees. This ensures that
    for certain values of `ell_comps` the angle does not jump from one boundary to another (e.g. without this check
    certain values of `ell_comps` return -1.0 degrees and others 179.0 degrees). This ensures that when error
    estimates are computed from samples of a lens model via marginalization, the calculation is not biased by the
    angle jumping between these two values.

    Parameters
    ----------
    ell_comps
        The elliptical components of the light or mass profile which are converted to an angle.
    """
    angle = xp.arctan2(ell_comps[0], ell_comps[1]) / 2
    angle *= 180.0 / xp.pi

    angle = xp.where(angle < -45, angle + 180, angle)

    fac = xp.sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2)
    if xp.__name__.startswith("jax"):
        import jax

        fac = jax.lax.min(fac, 0.999)
    else:  # NumPy
        fac = np.minimum(fac, 0.999)

    axis_ratio = (1 - fac) / (1 + fac)
    return axis_ratio, angle


def axis_ratio_from(ell_comps: Tuple[float, float], xp=np):
    """
    Returns the axis-ratio from input elliptical components e1 and e2 of a light or mass profile.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-axis_ratio)/(1+axis_ratio) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-axis_ratio)/(1+axis_ratio) * cos(2 * angle)

    The axis-ratio is therefore given by:

    axis_ratio = (1 - fac) / (1 + fac)
    angle = 0.5 * arctan(ell_comps[0] / ell_comps[1])

    where `fac = sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2).

    This function returns the axis-ratio.

    Parameters
    ----------
    ell_comps
        The elliptical components of the light or mass profile which are converted to an angle.
    """
    axis_ratio, angle = axis_ratio_and_angle_from(ell_comps=ell_comps, xp=xp)
    return axis_ratio


def angle_from(ell_comps: Tuple[float, float], xp=np) -> float:
    """
    Returns the position angle in degrees (-45 < angle < 135.0) from input elliptical components e1 and e2
    of a light or mass profile.

    The elliptical components of a light or mass profile are given by:

    elliptical_component_y = ell_comps[0] = (1-q)/(1+q) * sin(2 * angle)
    elliptical_component_x = ell_comps[1] = (1-q)/(1+q) * cos(2 * angle)

    axis_ratio = (1 - fac) / (1 + fac)
    angle = 0.5 * arctan(ell_comps[0] / ell_comps[1])

    where `fac = sqrt(ell_comps[1] ** 2 + ell_comps[0] ** 2).

    This function returns the angle in degrees.

    An additional check is performed which requires the angle is between -45 and 135 degrees. This ensures that
    for certain values of `ell_comps` the angle does not jump from one boundary to another (e.g. without this check
    certain values of `ell_comps` return -1.0 degrees and others 179.0 degrees).

    This ensures that when error estimates are computed from samples of a lens model via marginalization, the
    calculation is not biased by the angle jumping between these two values.

    Parameters
    ----------
    ell_comps
        The elliptical components of the light or mass profile which are converted to an angle.
    """
    axis_ratio, angle = axis_ratio_and_angle_from(ell_comps=ell_comps, xp=xp)
    return angle


def shear_gamma_1_2_from(magnitude: float, angle: float, xp=np) -> Tuple[float, float]:
    """
    Returns the shear gamma 1 and gamma 2 values an input shear magnitude and angle in degrees.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * xp.cos(2 * angle * xp.pi / 180.0)
    gamma_2 = magnitude * xp.sin(2 * angle * xp.pi / 180.0)

    Which are the values this function returns.

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = xp.sqrt(gamma_1**2 + gamma_2**2)
    angle = xp.arctan2(gamma_2, gamma_1) / 2 * 180.0 / xp.pi

    Parameters
    ----------
    axis_ratio
        Ratio of light profiles ellipse's minor and major axes (b/a).
    angle
        Rotation angle of light profile counter-clockwise from positive x-axis.
    """
    gamma_1 = magnitude * xp.cos(2 * angle * xp.pi / 180.0)
    gamma_2 = magnitude * xp.sin(2 * angle * xp.pi / 180.0)
    return (gamma_1, gamma_2)


def shear_magnitude_and_angle_from(
    gamma_1: float, gamma_2: float, xp=np
) -> Tuple[float, float]:
    """
    Returns the shear magnitude and angle in degrees from input shear gamma 1 and gamma 2 values.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * xp.cos(2 * angle * xp.pi / 180.0)
    gamma_2 = magnitude * xp.sin(2 * angle * xp.pi / 180.0)

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = xp.sqrt(gamma_1**2 + gamma_2**2)
    angle = xp.arctan2(gamma_2, gamma_1) / 2 * 180.0 / xp.pi

    Which are the values this function returns.

    Additional checks are performed which requires the angle is between -45 and 135 degrees. This ensures that
    for certain values of `gamma_1` and `gamma_2` the angle does not jump from one boundary to another (e.g. without
    this check certain values of `gamma_1` and `gamma_2` return -1.0 degrees and others 179.0 degrees).

    This ensures that when error estimates are computed from samples of a lens model via marginalization, the
    calculation is not biased by the angle jumping between these two values.

    Parameters
    ----------
    gamma_1
        The gamma 1 component of the shear.
    gamma_2
        The gamma 2 component of the shear.
    """
    angle = xp.arctan2(gamma_2, gamma_1) / 2 * 180.0 / xp.pi
    magnitude = xp.sqrt(gamma_1**2 + gamma_2**2)

    angle = xp.where(angle < 0, angle + 180.0, angle)
    angle = xp.where(
        (xp.abs(angle - 90.0) > 45.0) & (angle > 90.0), angle - 180.0, angle
    )

    return magnitude, angle


def shear_magnitude_from(gamma_1: float, gamma_2: float, xp=np) -> float:
    """
    Returns the shear magnitude and angle in degrees from input shear gamma 1 and gamma 2 values.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * xp.cos(2 * angle * xp.pi / 180.0)
    gamma_2 = magnitude * xp.sin(2 * angle * xp.pi / 180.0)

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = xp.sqrt(gamma_1**2 + gamma_2**2)
    angle = xp.arctan2(gamma_2, gamma_1) / 2 * 180.0 / xp.pi

    The magnitude value is what this function returns.

    Parameters
    ----------
    gamma_1
        The gamma 1 component of the shear.
    gamma_2
        The gamma 2 component of the shear.
    """
    magnitude, angle = shear_magnitude_and_angle_from(
        gamma_1=gamma_1, gamma_2=gamma_2, xp=xp
    )
    return magnitude


def shear_angle_from(gamma_1: float, gamma_2: float, xp=np) -> float:
    """
    Returns the shear magnitude and angle in degrees from input shear gamma 1 and gamma 2 values.

    The gamma 1 and gamma 2 components of a shear are given by:

    gamma_1 = magnitude * xp.cos(2 * angle * xp.pi / 180.0)
    gamma_2 = magnitude * xp.sin(2 * angle * xp.pi / 180.0)

    Converting from gamma 1 and gamma 2 to magnitude and angle is given by:

    magnitude = xp.sqrt(gamma_1**2 + gamma_2**2)
    angle = xp.arctan2(gamma_2, gamma_1) / 2 * 180.0 / xp.pi

    The angle value is what this function returns.

    Parameters
    ----------
    gamma_1
        The gamma 1 component of the shear.
    gamma_2
        The gamma 2 component of the shear.
    """
    magnitude, angle = shear_magnitude_and_angle_from(
        gamma_1=gamma_1, gamma_2=gamma_2, xp=xp
    )
    return angle


def multipole_k_m_and_phi_m_from(
    multipole_comps: Tuple[float, float], m: int, xp=np
) -> Tuple[float, float]:
    """
    Returns the multipole normalization value `k_m` and angle `phi` from the multipole component parameters.

    The normalization and angle are given by:

    .. math::
        \phi^{\rm mass}_m = \frac{1}{m} \arctan{\frac{\epsilon_{\rm 2}^{\rm mp}}{\epsilon_{\rm 1}^{\rm mp}}}, \, \,
        k^{\rm mass}_m = \sqrt{{\epsilon_{\rm 1}^{\rm mp}}^2 + {\epsilon_{\rm 2}^{\rm mp}}^2} \, .

    The conversion depends on the multipole order `m`, to ensure that all possible rotationally symmetric
    multiple mass profiles are available in the conversion for multiple components spanning -inf to inf.

    Additional checks are performed which requires the angle `phi_m` is between -45 and 135 degrees. This ensures that
    for certain multipole component values the angle does not jump from one boundary to another (e.g. without
    this check certain values of `gamma_1` and `gamma_2` return -1.0 degrees and others 179.0 degrees).

    This ensures that when error estimates are computed from samples of a lens model via marginalization, the
    calculation is not biased by the angle jumping between these two values.

    Parameters
    ----------
    multipole_comps
        The first and second components of the multipole.

    Returns
    -------
    The normalization and angle parameters of the multipole.
    """
    phi_m = (
        xp.arctan2(multipole_comps[0], multipole_comps[1]) * 180.0 / xp.pi / float(m)
    )
    k_m = xp.sqrt(multipole_comps[1] ** 2 + multipole_comps[0] ** 2)

    phi_m = xp.where(phi_m < -90.0 / m, phi_m + 360.0 / m, phi_m)

    return k_m, phi_m


def multipole_comps_from(
    k_m: float, phi_m: float, m: int, xp=np
) -> Tuple[float, float]:
    """
    Returns the multipole component parameters from their normalization value `k_m` and angle `phi`.

    .. math::
        \phi^{\rm mass}_m = \frac{1}{m} \arctan{\frac{\epsilon_{\rm 2}^{\rm mp}}{\epsilon_{\rm 1}^{\rm mp}}}, \, \,
        k^{\rm mass}_m = \sqrt{{\epsilon_{\rm 1}^{\rm mp}}^2 + {\epsilon_{\rm 2}^{\rm mp}}^2} \, .

    The conversion depends on the multipole order `m`, to ensure that all possible rotationally symmetric
    multiple mass profiles are available in the conversion for multiple components spanning -inf to inf.

    Parameters
    ----------
    k_m
        The magnitude of the multipole.
    phi_m
        The angle of the multipole.

    Returns
    -------
    The multipole component parameters.
    """
    from astropy import units

    multipole_comp_0 = k_m * xp.sin(phi_m * float(m) * units.deg.to(units.rad))
    multipole_comp_1 = k_m * xp.cos(phi_m * float(m) * units.deg.to(units.rad))

    return (multipole_comp_0, multipole_comp_1)

def shapelet_axis_ratio_and_phi_from(
    ell_comps: Tuple[float, float],
    xp=np,
) -> Tuple[float, float]:
    """
    Returns the elliptical axis-ratio `q` and position angle `phi` (in degrees) from the shapelet
    elliptical component parameters `ell_comps`.

    This conversion is intentionally identical in *spirit* to the `ell_comps` parameterization used
    throughout PyAutoGalaxy: the circular case corresponds to (0.0, 0.0) at the centre of parameter space,
    avoiding the sampling pathologies of directly sampling `(q, phi)` where the angle is undefined at `q=1`.

    For shapelets, these components define the *elliptical coordinate system* on which the basis functions
    are evaluated. This is geometric ellipticity (an isophote ellipse), therefore we always enforce the
    ellipse's 180 degree rotational symmetry by using an effective order m = 2 (and we do **not** accept
    `m` as an input).

    The conversion is:

    1) Convert components -> "ellipticity-like" amplitude and angle:

    .. math::
        e = \\sqrt{\\epsilon_1^2 + \\epsilon_2^2}

    .. math::
        \\phi = \\frac{1}{2} \\arctan2(\\epsilon_1, \\epsilon_2)

    2) Map amplitude -> axis-ratio using the standard stable relation:

    .. math::
        e = \\frac{1 - q}{1 + q} \\;\\;\\Rightarrow\\;\\;
        q = \\frac{1 - e}{1 + e}

    The returned `phi` is wrapped to prevent boundary hopping when computing marginalized error estimates
    from posterior samples. The wrapping enforces a continuous interval analogous to the multipole
    conversion logic, but with the fixed symmetry m = 2.

    Parameters
    ----------
    ell_comps
        The first and second components of the shapelet ellipticity. The circular limit is (0.0, 0.0).
        These are unconstrained and can span (-inf, inf) during sampling.
    xp
        The array library used for the calculation (e.g. `numpy` or `jax.numpy`).

    Returns
    -------
    axis_ratio
        The axis-ratio of the elliptical coordinate system, with 0 < q <= 1.
    phi
        The position angle in degrees, measured counter-clockwise from the positive x-axis.
    """
    eps_1, eps_2 = ell_comps

    # Ellipticity-like amplitude (0 at circular). Clip to keep q well-defined in (0, 1].
    e = xp.sqrt(eps_1 * eps_1 + eps_2 * eps_2)
    e = xp.clip(e, 0.0, 1.0 - 1e-12)

    axis_ratio = (1.0 - e) / (1.0 + e)

    # Fixed symmetry for ellipses: m = 2.
    phi = xp.arctan2(eps_1, eps_2) * 180.0 / xp.pi / 2.0

    # Wrap phi to a continuous interval to avoid boundary hopping in posteriors.
    # (Analogue of: phi_m = where(phi_m < -90/m, phi_m + 360/m, phi_m) with m=2.)
    phi = xp.where(phi < -45.0, phi + 180.0, phi)

    return axis_ratio, phi


def shapelet_ell_comps_from_axis_ratio_and_phi(
    axis_ratio: float,
    phi: float,
    xp=np,
) -> Tuple[float, float]:
    """
    Returns the shapelet elliptical component parameters `ell_comps` from an axis-ratio `q`
    and position angle `phi` (in degrees).

    This is the inverse of `shapelet_axis_ratio_and_phi_from` and uses the same fixed ellipse symmetry
    (effective order m = 2). The mapping is:

    1) Convert axis-ratio -> ellipticity-like amplitude:

    .. math::
        e = \\frac{1 - q}{1 + q}

    2) Convert amplitude and angle -> components:

    .. math::
        \\epsilon_1 = e \\, \\sin(2 \\phi)

    .. math::
        \\epsilon_2 = e \\, \\cos(2 \\phi)

    This ensures:

    .. math::
        \\phi = \\frac{1}{2} \\arctan2(\\epsilon_1, \\epsilon_2)

    Parameters
    ----------
    axis_ratio
        The axis-ratio of the elliptical coordinate system, with 0 < q <= 1.
    phi
        The position angle in degrees, measured counter-clockwise from the positive x-axis.
    xp
        The array library used for the calculation (e.g. `numpy` or `jax.numpy`).

    Returns
    -------
    ell_comps
        The first and second components of the shapelet ellipticity, where the circular limit is (0.0, 0.0).
    """
    axis_ratio = xp.clip(axis_ratio, 1e-12, 1.0)

    e = (1.0 - axis_ratio) / (1.0 + axis_ratio)

    # Fixed symmetry for ellipses: m = 2.
    ang = 2.0 * phi * xp.pi / 180.0

    eps_1 = e * xp.sin(ang)
    eps_2 = e * xp.cos(ang)

    return (eps_1, eps_2)
