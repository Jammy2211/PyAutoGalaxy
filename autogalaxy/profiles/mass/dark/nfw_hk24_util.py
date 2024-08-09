"""
Created on Wed Apr  3 15:07:27 2024

@author: felixvecchi
"""
import numpy as np
from astropy.cosmology import Planck15

cosmo = Planck15


def semi_major_axis_from(x1 : np.ndarray, x2 : np.ndarray, e : np.ndarray) -> np.ndarray:
    """
    Returns the semi-major axis of the ellipse at a given point.

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless
    x2
        Vertical coordinate, scaled by r_s, so unitless
    e
        Eccentricity.
    """
    return np.sqrt(x1 ** 2 + x2 ** 2 / (1 - e ** 2))


def capital_F_from(chi : np.ndarray) -> np.ndarray:
    """
    Equation 16 from Heyrovský & Karamazov.

    Parameters
    ----------
    chi
        Dimenionless radial coordinate.

    Returns
    -------
    F(chi)
    """
    F = np.zeros(chi.shape)

    root_min = np.sqrt(1 - chi[chi < 1] ** 2)
    F[chi < 1] = np.arctanh(root_min) / root_min
    F[chi == 1] = 1
    root_plus = np.sqrt(chi[chi > 1] ** 2 - 1)
    F[chi > 1] = np.arctan(root_plus) / root_plus
    return F


def kappa_from(k_s : float, a : np.ndarray) -> np.ndarray:
    """
    Equation 16 from Heyrovský & Karamazov.

    Parameters
    ----------
    k_s
        Halo convergence parameter.
    a
        Semi major axis scaled by a_scale.

    Returns
    -------
    Convergence as a function of a
    """
    F = capital_F_from(a)
    kappa = 2 * k_s * (1 - F) / (a ** 2 - 1)
    kappa[a == 1] = 2 / 3 * k_s
    return kappa


def small_f_1(x1 : np.ndarray, x2 : np.ndarray, e : float) -> np.ndarray:
    """
    Equation 32 HK+24

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless
    x2
        Vertical coordinate, scaled by r_s, so unitless
    e
        Eccentricity.

    Returns
    -------
    f_1
    """
    a = semi_major_axis_from(x1, x2, e)
    F = capital_F_from(a)
    f1 = (1 - e**2) ** (-1 / 2) * F
    return f1


def small_f_2(x1 : np.ndarray, x2 : np.ndarray, e : float) -> np.ndarray:
    """
    Equation 32 HK+24

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless
    x2
        Vertical coordinate, scaled by r_s, so unitless
    e
        Eccentricity.

    Returns
    -------
    f_3

    """
    norm = np.sqrt(x1**2 + x2**2)
    f2 = np.log(norm / (1 + np.sqrt(1 - e**2)))
    return f2


def small_f_3(x1 : np.ndarray, x2 : np.ndarray, e : float) -> np.ndarray:
    """
    Equation 32 HK+24

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless
    x2
        Vertical coordinate, scaled by r_s, so unitless
    e
        Eccentricity.

    Returns
    -------
    f_3

    """
    root = np.sqrt(1 - e**2)
    f3 = np.arctan(x1 * x2 * (1 - root) / (x1**2 * root + x2**2))
    return f3


def small_f_0(x1 : np.ndarray, x2 : np.ndarray, e : float) -> np.ndarray:
    """
    Equation 37 HK+24

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless
    x2
        Vertical coordinate, scaled by r_s, so unitless
    e
        Eccentricity.

    Returns
    -------
    f_0

    """
    a = semi_major_axis_from(x1, x2, e)
    F = capital_F_from(a)
    pre_factor = 1 / (2 * np.sqrt(1 - e**2))
    nominator = x1**2 + x2**2 + e**2 - 2 + (1 - e**2 * x1**2) * F
    denominator = 1 - x1**2 - x2**2 / (1 - e**2)

    f0 = 1 + pre_factor * (nominator / denominator)

    return f0


def gamma1(x1 : np.ndarray, x2 : np.ndarray, e : float, k_s : float) -> np.ndarray:
    """
    Equation 35 HK+24

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless.
    x2
        Vertical coordinate, scaled by r_s, so unitless.
    e
        Eccentricity.
    k_s
        Halo convergence parameter.

    Returns
    -------
    First component of shear ('+' shape)

    """
    full_pre_factor = (
        4
        * k_s
        * np.sqrt(1 - e**2)
        / (((x1 - e) ** 2 + x2**2) ** 2 + ((x1 + e) ** 2 + x2**2) ** 2)
    )

    pre_f0 = (
        ((x1 - e) ** 2 + x2**2)
        * ((x1 + e) ** 2 + x2**2)
        * (x1**2 - x2**2 - e**2)
    )

    pre_f1 = (
        2
        * e**2
        * (x1**2 - 1)
        * ((x1**2 - x2**2 - e**2) ** 2 - 4 * x1**2 * x2**2)
        - (x1**2 - x2**2 - e**2) ** 3 * (3 + e**2) / 2
        + 6 * x1**2 * x2**2 * (e**2 - 1) * (x1**2 - x2**2 - e**2)
    )

    pre_f2 = -(
        (x1**2 - x2**2 - e**2) * ((x1**2 + x2**2) ** 2 - e**4)
        - 8 * e**2 * x1**2 * x2**2
    )

    pre_f3 = (
        2
        * x1
        * x2
        * ((x1**2 + x2**2 + e**2) ** 2 - 4 * e**2 * (x2**2 + e**2))
    )

    f0 = small_f_0(x1, x2, e)
    f1 = small_f_1(x1, x2, e)
    f2 = small_f_2(x1, x2, e)
    f3 = small_f_3(x1, x2, e)

    g1 = full_pre_factor * (pre_f0 * f0 + pre_f1 * f1 + pre_f2 * f2 + pre_f3 * f3)
    return g1


def gamma2(x1 : np.ndarray, x2 : np.ndarray, e : float, k_s : float) -> np.ndarray:
    """
    Equation 36 HK+24

    Parameters
    ----------
    x1
        Horizontal coordinate, scaled by r_s, so unitless.
    x2
        Vertical coordinate, scaled by r_s, so unitless.
    e
        Eccentricity.
    k_s
        Halo convergence parameter.

    Returns
    -------
    Second component of shear ('x' shape)
    """
    full_pre_factor = (
        4
        * k_s
        * np.sqrt(1 - e**2)
        / (((x1 - e) ** 2 + x2**2) ** 2 + ((x1 + e) ** 2 + x2**2) ** 2)
    )

    pre_f0 = 2 * x1 * x2 * ((x1 - e) ** 2 + x2**2) * ((x1 + e) ** 2 + x2**2)

    pre_f1 = (
        x1
        * x2
        * (
            (x1**2 + x2**2 + e**2)
            * (
                (5 * e**2 - 3) * x1**2
                - 3 * (1 + e**2) * x2**2
                + (5 - 3 * e**2) * e**2
            )
            - 4 * e**2 * x1**2 * (1 + e**2)
        )
    )

    pre_f2 = (
        -2
        * x1
        * x2
        * ((x1**2 + x2**2 + e**2) ** 2 - 4 * e**2 * (x2**2 + e**2))
    )

    pre_f3 = -(
        (x1**2 - x2**2 - e**2) * ((x1**2 + x2**2) ** 2 - e**4)
        - 8 * e**2 * x1**2 * x2**2
    )

    f0 = small_f_0(x1, x2, e)
    f1 = small_f_1(x1, x2, e)
    f2 = small_f_2(x1, x2, e)
    f3 = small_f_3(x1, x2, e)

    g2 = full_pre_factor * (pre_f0 * f0 + pre_f1 * f1 + pre_f2 * f2 + pre_f3 * f3)

    return g2
