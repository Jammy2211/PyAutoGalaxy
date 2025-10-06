from __future__ import division, print_function
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():
    lp = ag.lp.Exponential(
        ell_comps=(0.0, 0.333333), intensity=3.0, effective_radius=2.0
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert image == pytest.approx(4.9047, 1e-3)

    lp = ag.lp.Exponential(
        ell_comps=(0.0, -0.333333), intensity=2.0, effective_radius=3.0
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(4.8566, 1e-3)

    lp = ag.lp.Exponential(
        ell_comps=(0.0, -0.333333), intensity=4.0, effective_radius=3.0
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(2.0 * 4.8566, 1e-3)

    value = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert value == pytest.approx(2.0 * 4.8566, 1e-3)

    elliptical = ag.lp.Exponential(
        ell_comps=(0.0, 0.0), intensity=3.0, effective_radius=2.0
    )

    spherical = ag.lp.ExponentialSph(intensity=3.0, effective_radius=2.0)

    image_elliptical = elliptical.image_2d_from(grid=grid)
    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical.array == pytest.approx(image_spherical.array, 1.0e-4)
