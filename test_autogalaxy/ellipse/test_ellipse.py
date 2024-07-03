import numpy as np
import pytest

import autogalaxy as ag


def test__circular_radius():
    ellipse = ag.Ellipse(major_axis=1.0)

    assert ellipse.circular_radius == pytest.approx(6.283185307, 1.0e-4)

    ellipse = ag.Ellipse(major_axis=0.5)

    assert ellipse.circular_radius == pytest.approx(3.141592654, 1.0e-4)


def test__ellipticity():
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0))

    assert ellipse.eccentricity == 0.0

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.5))

    assert ellipse.eccentricity == pytest.approx(0.5, 1.0e-4)

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.0))

    assert ellipse.eccentricity == pytest.approx(0.5, 1.0e-4)

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5))

    assert ellipse.eccentricity == pytest.approx(np.sqrt(2) / 2.0, 1.0e-4)


def test__minor_axis_from():
    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    assert ellipse.minor_axis == 1.0

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=0.5)

    assert ellipse.minor_axis == 0.5

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.5), major_axis=1.0)

    assert ellipse.minor_axis == pytest.approx(0.866025403, 1.0e-4)
