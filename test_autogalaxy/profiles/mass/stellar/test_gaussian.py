import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_2d_via_analytic_from():

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.05263),
        intensity=1.0,
        sigma=3.0,
        mass_to_light_ratio=1.0,
    )

    deflections = gaussian.deflections_2d_via_analytic_from(grid=np.array([[1.0, 0.0]]))

    assert deflections[0, 0] == pytest.approx(1.024423, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.0, 1.0e-4)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = gaussian.deflections_2d_via_analytic_from(grid=np.array([[0.5, 0.2]]))

    assert deflections[0, 0] == pytest.approx(0.554062, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.177336, 1.0e-4)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )

    deflections = gaussian.deflections_2d_via_analytic_from(grid=np.array([[0.5, 0.2]]))

    assert deflections[0, 0] == pytest.approx(1.108125, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.35467, 1.0e-4)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.111111),
        intensity=2.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = gaussian.deflections_2d_via_analytic_from(grid=np.array([[0.5, 0.2]]))

    assert deflections[0, 0] == pytest.approx(1.10812, 1.0e-4)
    assert deflections[0, 1] == pytest.approx(0.35467, 1.0e-4)


def test__deflections_2d_via_integral_from():

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.05263),
        intensity=1.0,
        sigma=3.0,
        mass_to_light_ratio=1.0,
    )

    deflections = gaussian.deflections_2d_via_integral_from(grid=np.array([[1.0, 0.0]]))
    deflections_via_analytic = gaussian.deflections_2d_via_analytic_from(
        grid=np.array([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_analytic, 1.0e-3)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = gaussian.deflections_2d_via_integral_from(grid=np.array([[0.5, 0.2]]))
    deflections_via_analytic = gaussian.deflections_2d_via_analytic_from(
        grid=np.array([[0.5, 0.2]])
    )

    assert deflections == pytest.approx(deflections_via_analytic, 1.0e-3)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.111111),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )

    deflections = gaussian.deflections_2d_via_integral_from(grid=np.array([[0.5, 0.2]]))
    deflections_via_analytic = gaussian.deflections_2d_via_analytic_from(
        grid=np.array([[0.5, 0.2]])
    )

    assert deflections == pytest.approx(deflections_via_analytic, 1.0e-3)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.111111),
        intensity=2.0,
        sigma=5.0,
        mass_to_light_ratio=1.0,
    )

    deflections = gaussian.deflections_2d_via_integral_from(grid=np.array([[0.5, 0.2]]))
    deflections_via_analytic = gaussian.deflections_2d_via_analytic_from(
        grid=np.array([[0.5, 0.2]])
    )

    assert deflections == pytest.approx(deflections_via_analytic, 1.0e-3)


def test__deflections_yx_2d_from():

    gaussian = ag.mp.Gaussian()

    deflections = gaussian.deflections_yx_2d_from(grid=np.array([[1.0, 0.0]]))
    deflections_via_integral = gaussian.deflections_2d_via_analytic_from(
        grid=np.array([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral, 1.0e-4)


def test__convergence_2d_from():
    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=1.0,
        mass_to_light_ratio=1.0,
    )

    convergence = gaussian.convergence_2d_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.60653, 1e-2)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.0),
        intensity=1.0,
        sigma=1.0,
        mass_to_light_ratio=2.0,
    )

    convergence = gaussian.convergence_2d_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(2.0 * 0.60653, 1e-2)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.333333),
        intensity=2.0,
        sigma=3.0,
        mass_to_light_ratio=4.0,
    )

    convergence = gaussian.convergence_2d_from(grid=np.array([[0.0, 1.0]]))

    assert convergence == pytest.approx(7.88965, 1e-2)


def test__intensity_and_convergence_match_for_mass_light_ratio_1():

    gaussian_light_profile = ag.lp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.333333),
        intensity=2.0,
        sigma=3.0,
    )

    gaussian_mass_profile = ag.mp.Gaussian(
        centre=(0.0, 0.0),
        elliptical_comps=(0.0, 0.333333),
        intensity=2.0,
        sigma=3.0,
        mass_to_light_ratio=1.0,
    )

    intensity = gaussian_light_profile.image_2d_from(grid=np.array([[1.0, 0.0]]))
    convergence = gaussian_mass_profile.convergence_2d_from(grid=np.array([[1.0, 0.0]]))

    assert (intensity == convergence).all()


def test__image_2d_via_radii_from__correct_value():
    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=1.0
    )

    intensity = gaussian.image_2d_via_radii_from(grid_radii=1.0)

    assert intensity == pytest.approx(0.60653, 1e-2)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=2.0, sigma=1.0
    )

    intensity = gaussian.image_2d_via_radii_from(grid_radii=1.0)

    assert intensity == pytest.approx(2.0 * 0.60653, 1e-2)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
    )

    intensity = gaussian.image_2d_via_radii_from(grid_radii=1.0)

    assert intensity == pytest.approx(0.882496, 1e-2)

    gaussian = ag.mp.Gaussian(
        centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), intensity=1.0, sigma=2.0
    )

    intensity = gaussian.image_2d_via_radii_from(grid_radii=3.0)

    assert intensity == pytest.approx(0.32465, 1e-2)
