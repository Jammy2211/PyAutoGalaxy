import numpy as np


def elliptical_comps_from(axis_ratio, angle):
    """
    Convert an input axis ratio (0.0 > q > 1.0) and rotation position angle defined counter clockwise from the
    positive x-axis(0.0 > angle > 180) to the (y,x) ellipitical components e1 and e2.

    Parameters
    ----------
    axis_ratio
            Ratio of light profiles ellipse's minor and major axes (b/a).
    angle
            Rotation angle of light profile counter-clockwise from positive x-axis.
    """
    angle *= np.pi / 180.0
    fac = (1 - axis_ratio) / (1 + axis_ratio)
    ellip_y = fac * np.sin(2 * angle)
    ellip_x = fac * np.cos(2 * angle)
    return (ellip_y, ellip_x)


def axis_ratio_and_angle_from(elliptical_comps):
    """
    Convert the ellipitical components e1 and e2 to an axis ratio (0.0 > q > 1.0) and rotation position angle
    defined counter clockwise from the positive x-axis(0.0 > angle > 180) to .

    Parameters
    ----------
    elliptical_comps : (float, float)
        The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
    """
    angle = np.arctan2(elliptical_comps[0], elliptical_comps[1]) / 2
    angle *= 180.0 / np.pi
    fac = np.sqrt(elliptical_comps[1] ** 2 + elliptical_comps[0] ** 2)
    if fac > 0.999:
        fac = 0.999  # avoid unphysical solution
    # if fac > 1: print('unphysical e1,e2')
    axis_ratio = (1 - fac) / (1 + fac)
    return axis_ratio, angle


def axis_ratio_from(elliptical_comps):
    """
    Convert the ellipitical components e1 and e2 to an axis ratio (0.0 > q > 1.0) and rotation position angle
    defined counter clockwise from the positive x-axis(0.0 > angle > 180) to .

    Parameters
    ----------
    elliptical_comps : (float, float)
        The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
    """
    axis_ratio, angle = axis_ratio_and_angle_from(elliptical_comps=elliptical_comps)
    return axis_ratio


def angle_from(elliptical_comps):
    """
    Convert the ellipitical components e1 and e2 to an axis ratio (0.0 > q > 1.0) and rotation position angle
    defined counter clockwise from the positive x-axis(0.0 > angle > 180) to .

    Parameters
    ----------
    elliptical_comps : (float, float)
        The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
    """
    axis_ratio, angle = axis_ratio_and_angle_from(elliptical_comps=elliptical_comps)
    return angle


def shear_elliptical_comps_from(magnitude, angle):
    """
    :param angle: angel
    :param magnitude: ellipticity
    :return:
    """
    ellip_y = magnitude * np.sin(2 * angle * np.pi / 180.0)
    ellip_x = magnitude * np.cos(2 * angle * np.pi / 180.0)
    return (ellip_y, ellip_x)


def shear_magnitude_and_angle_from(elliptical_comps):
    """
    :param e1: ellipticity component
    :param e2: ellipticity component
    :return: angle and abs value of ellipticity
    """
    angle = np.arctan2(elliptical_comps[0], elliptical_comps[1]) / 2 * 180.0 / np.pi
    magnitude = np.sqrt(elliptical_comps[1] ** 2 + elliptical_comps[0] ** 2)
    if angle < 0:
        return magnitude, angle + 180.0
    return magnitude, angle


def shear_magnitude_from(elliptical_comps):
    """
    :param e1: ellipticity component
    :param e2: ellipticity component
    :return: angle and abs value of ellipticity
    """
    magnitude, angle = shear_magnitude_and_angle_from(elliptical_comps=elliptical_comps)
    return magnitude


def shear_angle_from(elliptical_comps):
    """
    :param e1: ellipticity component
    :param e2: ellipticity component
    :return: angle and abs value of ellipticity
    """
    magnitude, angle = shear_magnitude_and_angle_from(elliptical_comps=elliptical_comps)
    return angle
