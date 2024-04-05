from __future__ import division, print_function
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():
    lp = ag.lp.DevVaucouleurs(
        ell_comps=(0.0, 0.333333), intensity=3.0, effective_radius=2.0
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert image == pytest.approx(5.6697, 1e-3)

    lp = ag.lp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333), intensity=2.0, effective_radius=3.0
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(7.4455, 1e-3)

    lp = ag.lp.DevVaucouleurs(
        ell_comps=(0.0, -0.333333), intensity=4.0, effective_radius=3.0
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(2.0 * 7.4455, 1e-3)

    value = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert value == pytest.approx(2.0 * 7.4455, 1e-3)

    elliptical = ag.lp.DevVaucouleurs(
        ell_comps=(0.0, 0.0), intensity=3.0, effective_radius=2.0
    )

    spherical = ag.lp.DevVaucouleursSph(intensity=3.0, effective_radius=2.0)

    image_elliptical = elliptical.image_2d_from(grid=grid)

    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)
