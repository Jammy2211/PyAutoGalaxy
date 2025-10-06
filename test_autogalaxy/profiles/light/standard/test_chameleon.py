from __future__ import division, print_function
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():
    lp = ag.lp.Chameleon(
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(0.018605, 1e-3)

    lp = ag.lp.Chameleon(
        ell_comps=(0.5, 0.0),
        intensity=3.0,
        core_radius_0=0.2,
        core_radius_1=0.4,
    )
    # 3.0 * exp(-3.67206544592 * (1,5/2.0) ** (1.0 / 2.0)) - 1) = 0.351797

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.5]]))

    assert image == pytest.approx(0.0078149, 1e-3)

    lp = ag.lp.Chameleon(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        core_radius_0=0.2,
        core_radius_1=0.4,
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert image == pytest.approx(0.024993, 1e-3)

    elliptical = ag.lp.Chameleon(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        core_radius_0=0.2,
        core_radius_1=0.4,
    )

    spherical = ag.lp.ChameleonSph(intensity=3.0, core_radius_0=0.2, core_radius_1=0.4)

    image_elliptical = elliptical.image_2d_from(grid=grid)

    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical.array == pytest.approx(image_spherical.array, 1.0e-4)
