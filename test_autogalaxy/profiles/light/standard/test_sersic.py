from __future__ import division, print_function
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():
    lp = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert image == pytest.approx(0.351797, 1e-3)

    lp = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
    )
    # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.5, 0.0]]))

    assert image == pytest.approx(4.90657319276, 1e-3)

    lp = ag.lp.Sersic(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert image == pytest.approx(5.38066670129, 1e-3)

    elliptical = ag.lp.Sersic(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
    )

    spherical = ag.lp.SersicSph(intensity=3.0, effective_radius=2.0, sersic_index=2.0)

    image_elliptical = elliptical.image_2d_from(grid=grid)

    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)
