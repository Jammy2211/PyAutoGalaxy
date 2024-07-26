import pytest

import autogalaxy as ag


def test__points_perturbed_from():
    pixel_scale = 1.0

    ellipse = ag.Ellipse(major_axis=1.0)

    points = ellipse.points_from_major_axis_from(pixel_scale=pixel_scale)

    multipole = ag.EllipseMultipole(m=4, multipole_comps=(0.0, 0.0))

    points_perturbed = multipole.points_perturbed_from(
        pixel_scale=pixel_scale, points=points, ellipse=ellipse
    )

    assert points_perturbed == pytest.approx(points, 1.0e-4)

    multipole = ag.EllipseMultipole(m=4, multipole_comps=(0.1, 0.2))

    points_perturbed = multipole.points_perturbed_from(
        pixel_scale=pixel_scale, points=points, ellipse=ellipse
    )

    assert points_perturbed[1, 0] == pytest.approx(-0.982728, 1.0e-4)
    assert points_perturbed[1, 1] == pytest.approx(0.298726, 1.0e-4)
