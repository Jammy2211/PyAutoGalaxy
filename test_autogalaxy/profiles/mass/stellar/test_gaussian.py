import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_2d_via_analytic_from():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.05263),
        intensity=1.0,
        sigma=3.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections[0, 0] == pytest.approx(1.024423, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )

    assert deflections[0, 0] == pytest.approx(0.554062, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.177336, 1.0e-4)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )

    assert deflections[0, 0] == pytest.approx(1.108125, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.35467, 1.0e-4)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=2.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )

    assert deflections[0, 0] == pytest.approx(1.10812, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.35467, 1.0e-4)


def test__deflections_2d_via_integral_from():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.05263),
        intensity=1.0,
        sigma=3.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )
    deflections_via_analytic = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_analytic, 1.0e-3)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )
    deflections_via_analytic = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )

    assert deflections == pytest.approx(deflections_via_analytic, 1.0e-3)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )

    deflections = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )
    deflections_via_analytic = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )

    assert deflections == pytest.approx(deflections_via_analytic, 1.0e-3)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.111111),
        intensity=2.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )
    deflections_via_analytic = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[0.5, 0.2]])
    )

    assert deflections == pytest.approx(deflections_via_analytic, 1.0e-3)


def test__deflections_yx_2d_from():
    mp = ag.mp.Gaussian()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_analytic_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral, 1.0e-4)


def test__convergence_2d_from():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=1.0,
        mass_to_light_ratio=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.60653, 1e-2)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=1.0,
        mass_to_light_ratio=2.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 0.60653, 1e-2)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=2.0,
        sigma=3.0,
        mass_to_light_ratio=4.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(7.88965, 1e-2)


def test__intensity_and_convergence_match_for_mass_light_ratio_1():
    lp = ag.lp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=2.0,
        sigma=3.0,
    )

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.333333),
        intensity=2.0,
        sigma=3.0,
        mass_to_light_ratio=1.0,
    )

    intensity = lp.image_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert (intensity == convergence).all()


def test__image_2d_via_radii_from__correct_value():
    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
    )

    intensity = mp.image_2d_via_radii_from(grid_radii=1.0)

    assert intensity == pytest.approx(0.60653, 1e-2)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=2.0, sigma=1.0
    )

    intensity = mp.image_2d_via_radii_from(grid_radii=1.0)

    assert intensity == pytest.approx(2.0 * 0.60653, 1e-2)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
    )

    intensity = mp.image_2d_via_radii_from(grid_radii=1.0)

    assert intensity == pytest.approx(0.882496, 1e-2)

    mp = ag.mp.Gaussian(
        centre=(0.0, 0.0), ell_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
    )

    intensity = mp.image_2d_via_radii_from(grid_radii=3.0)

    assert intensity == pytest.approx(0.32465, 1e-2)
