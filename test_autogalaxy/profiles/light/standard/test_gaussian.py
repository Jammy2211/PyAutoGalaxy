from __future__ import division, print_function
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__image_2d_from():
    lp = ag.lp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(0.60653, 1e-2)

    lp = ag.lp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=2.0, sigma=1.0
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(2.0 * 0.60653, 1e-2)

    lp = ag.lp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert image == pytest.approx(0.88249, 1e-2)

    lp = ag.lp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
    )

    image = lp.image_2d_from(grid=ag.Grid2DIrregular([[0.0, 3.0]]))

    assert image == pytest.approx(0.3246, 1e-2)

    elliptical = ag.lp.Gaussian(ell_comps=(0.0, 0.0), intensity=3.0, sigma=2.0)
    spherical = ag.lp.GaussianSph(intensity=3.0, sigma=2.0)

    image_elliptical = elliptical.image_2d_from(grid=grid)
    image_spherical = spherical.image_2d_from(grid=grid)

    assert image_elliptical == pytest.approx(image_spherical, 1.0e-4)
