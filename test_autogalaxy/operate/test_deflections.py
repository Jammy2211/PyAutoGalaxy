import numpy as np
import pytest

from skimage import measure

import autogalaxy as ag

from autogalaxy.operate.deflections import grid_scaled_2d_for_marching_squares_from


def critical_curve_via_magnification_from(mass_profile, grid):
    magnification = mass_profile.magnification_2d_from(grid=grid)

    inverse_magnification = 1 / magnification

    critical_curves_indices = measure.find_contours(
        np.array(inverse_magnification.native), 0
    )

    no_critical_curves = len(critical_curves_indices)
    contours = []
    critical_curves = []

    for jj in np.arange(no_critical_curves):
        contours.append(critical_curves_indices[jj])
        contour_x, contour_y = contours[jj].T
        pixel_coord = np.stack((contour_x, contour_y), axis=-1)

        critical_curve = grid_scaled_2d_for_marching_squares_from(
            grid_pixels_2d=pixel_coord,
            shape_native=magnification.shape_native,
            mask=grid.mask,
        )

        critical_curves.append(critical_curve)

    return critical_curves


def caustics_via_magnification_from(mass_profile, grid):
    caustics = []

    critical_curves = critical_curve_via_magnification_from(
        mass_profile=mass_profile, grid=grid
    )

    for i in range(len(critical_curves)):
        critical_curve = critical_curves[i]

        deflections_1d = mass_profile.deflections_yx_2d_from(grid=critical_curve)

        caustic = critical_curve - deflections_1d

        caustics.append(caustic)

    return caustics


def test__hessian_from():
    grid = ag.Grid2DIrregular(values=[(0.5, 0.5), (1.0, 1.0)])

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    hessian_yy, hessian_xy, hessian_yx, hessian_xx = mp.hessian_from(grid=grid)

    assert hessian_yy == pytest.approx(np.array([1.3883822, 0.694127]), 1.0e-4)
    assert hessian_xy == pytest.approx(np.array([-1.388124, -0.694094]), 1.0e-4)
    assert hessian_yx == pytest.approx(np.array([-1.388165, -0.694099]), 1.0e-4)
    assert hessian_xx == pytest.approx(np.array([1.3883824, 0.694127]), 1.0e-4)

    grid = ag.Grid2DIrregular(values=[(1.0, 0.0), (0.0, 1.0)])

    hessian_yy, hessian_xy, hessian_yx, hessian_xx = mp.hessian_from(grid=grid)

    assert hessian_yy == pytest.approx(np.array([0.0, 1.777699]), 1.0e-4)
    assert hessian_xy == pytest.approx(np.array([0.0, 0.0]), 1.0e-4)
    assert hessian_yx == pytest.approx(np.array([0.0, 0.0]), 1.0e-4)
    assert hessian_xx == pytest.approx(np.array([2.22209, 0.0]), 1.0e-4)


def test__convergence_2d_via_hessian_from():
    buffer = 0.0001
    grid = ag.Grid2DIrregular(
        values=[(1.075, -0.125), (-0.875, -0.075), (-0.925, -0.075), (0.075, 0.925)]
    )

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.001, 0.001), einstein_radius=1.0
    )

    convergence = mp.convergence_2d_via_hessian_from(grid=grid, buffer=buffer)

    assert convergence.in_list[0] == pytest.approx(0.461447, 1.0e-4)
    assert convergence.in_list[1] == pytest.approx(0.568875, 1.0e-4)
    assert convergence.in_list[2] == pytest.approx(0.538326, 1.0e-4)
    assert convergence.in_list[3] == pytest.approx(0.539390, 1.0e-4)

    mp = ag.mp.Isothermal(centre=(0.0, 0.0), ell_comps=(0.3, 0.4), einstein_radius=1.5)

    convergence = mp.convergence_2d_via_hessian_from(grid=grid, buffer=buffer)

    assert convergence.in_list[0] == pytest.approx(0.35313, 1.0e-4)
    assert convergence.in_list[1] == pytest.approx(0.46030, 1.0e-4)
    assert convergence.in_list[2] == pytest.approx(0.43484, 1.0e-4)
    assert convergence.in_list[3] == pytest.approx(1.00492, 1.0e-4)


def test__magnification_2d_via_hessian_from():
    grid = ag.Grid2DIrregular(values=[(0.5, 0.5), (1.0, 1.0)])

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    magnification = mp.magnification_2d_via_hessian_from(grid=grid)

    assert magnification.in_list[0] == pytest.approx(-0.56303, 1.0e-4)
    assert magnification.in_list[1] == pytest.approx(-2.57591, 1.0e-4)


def test__magnification_2d_from__compare_eigen_values_and_determinant():
    grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    magnification_via_determinant = mp.magnification_2d_from(grid=grid)
    tangential_eigen_value = mp.tangential_eigen_value_from(grid=grid)

    radal_eigen_value = mp.radial_eigen_value_from(grid=grid)
    magnification_via_eigen_values = 1 / (tangential_eigen_value * radal_eigen_value)

    mean_error = np.mean(
        magnification_via_determinant.slim - magnification_via_eigen_values.slim
    )

    assert mean_error < 1e-4


def test__magnification_2d_from__compare_determinant_and_convergence_and_shear():
    grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    magnification_via_determinant = mp.magnification_2d_from(grid=grid)

    convergence = mp.convergence_2d_via_jacobian_from(grid=grid)
    shear = mp.shear_yx_2d_via_jacobian_from(grid=grid)

    magnification_via_convergence_and_shear = 1 / (
        (1 - convergence) ** 2 - shear.magnitudes**2
    )

    mean_error = np.mean(
        magnification_via_determinant.slim
        - magnification_via_convergence_and_shear.slim
    )

    assert mean_error < 1e-4


def test__tangential_critical_curve_list_from():
    grid = ag.Grid2D.uniform(shape_native=(15, 15), pixel_scales=0.3)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    tangential_critical_curve_list = mp.tangential_critical_curve_list_from(grid=grid)

    x_critical_tangential, y_critical_tangential = (
        tangential_critical_curve_list[0][:, 1],
        tangential_critical_curve_list[0][:, 0],
    )

    assert np.mean(
        x_critical_tangential**2 + y_critical_tangential**2
    ) == pytest.approx(mp.einstein_radius**2, 5e-1)

    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    tangential_critical_curve_list = mp.tangential_critical_curve_list_from(grid=grid)

    y_centre = np.mean(tangential_critical_curve_list[0][:, 0])
    x_centre = np.mean(tangential_critical_curve_list[0][:, 1])

    assert -0.03 < y_centre < 0.03
    assert -0.03 < x_centre < 0.03

    mp = ag.mp.IsothermalSph(centre=(0.5, 1.0), einstein_radius=2.0)

    tangential_critical_curve_list = mp.tangential_critical_curve_list_from(grid=grid)

    y_centre = np.mean(tangential_critical_curve_list[0][:, 0])
    x_centre = np.mean(tangential_critical_curve_list[0][:, 1])

    assert 0.47 < y_centre < 0.53
    assert 0.97 < x_centre < 1.03


def test__tangential_critical_curve_list_from__compare_via_magnification():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=2, ell_comps=(0.109423, -0.019294)
    )

    tangential_critical_curve_via_magnification = critical_curve_via_magnification_from(
        mass_profile=mp, grid=grid
    )[0]

    tangential_critical_curve_list = mp.tangential_critical_curve_list_from(
        grid=grid, pixel_scale=0.2
    )

    assert tangential_critical_curve_list[0] == pytest.approx(
        tangential_critical_curve_via_magnification, 5e-1
    )

    tangential_critical_curve_via_magnification = critical_curve_via_magnification_from(
        mass_profile=mp, grid=grid
    )[0]

    tangential_critical_curve_list = mp.tangential_critical_curve_list_from(
        grid=grid, pixel_scale=0.2
    )

    assert tangential_critical_curve_list[0] == pytest.approx(
        tangential_critical_curve_via_magnification, 5e-1
    )


def test__radial_critical_curve_list_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    radial_critical_curve_list = mp.radial_critical_curve_list_from(grid=grid)

    y_centre = np.mean(radial_critical_curve_list[0][:, 0])
    x_centre = np.mean(radial_critical_curve_list[0][:, 1])

    assert -0.05 < y_centre < 0.05
    assert -0.05 < x_centre < 0.05

    mp = ag.mp.IsothermalSph(centre=(0.5, 1.0), einstein_radius=2.0)

    radial_critical_curve_list = mp.radial_critical_curve_list_from(grid=grid)

    y_centre = np.mean(radial_critical_curve_list[0][:, 0])
    x_centre = np.mean(radial_critical_curve_list[0][:, 1])

    assert 0.45 < y_centre < 0.55
    assert 0.95 < x_centre < 1.05


def test__radial_critical_curve_list_from__compare_via_magnification():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=2, ell_comps=(0.109423, -0.019294)
    )

    critical_curve_radial_via_magnification = critical_curve_via_magnification_from(
        mass_profile=mp, grid=grid
    )[1]

    radial_critical_curve_list = mp.radial_critical_curve_list_from(grid=grid)

    assert sum(critical_curve_radial_via_magnification) == pytest.approx(
        sum(radial_critical_curve_list[0]), abs=0.7
    )


def test__tangential_caustic_list_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    tangential_caustic_list = mp.tangential_caustic_list_from(grid=grid)

    y_centre = np.mean(tangential_caustic_list[0][:, 0])
    x_centre = np.mean(tangential_caustic_list[0][:, 1])

    assert -0.03 < y_centre < 0.03
    assert -0.03 < x_centre < 0.03

    mp = ag.mp.IsothermalSph(centre=(0.5, 1.0), einstein_radius=2.0)

    tangential_caustic_list = mp.tangential_caustic_list_from(grid=grid)

    y_centre = np.mean(tangential_caustic_list[0][:, 0])
    x_centre = np.mean(tangential_caustic_list[0][:, 1])

    assert 0.47 < y_centre < 0.53
    assert 0.97 < x_centre < 1.03


def test__tangential_caustic_list_from___compare_via_magnification():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=2, ell_comps=(0.109423, -0.019294)
    )

    tangential_caustic_via_magnification = caustics_via_magnification_from(
        mass_profile=mp, grid=grid
    )[0]

    tangential_caustic_list = mp.tangential_caustic_list_from(
        grid=grid, pixel_scale=0.2
    )

    assert sum(tangential_caustic_list[0]) == pytest.approx(
        sum(tangential_caustic_via_magnification), 5e-1
    )


def test__radial_caustic_list_from():
    grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    radial_caustic_list = mp.radial_caustic_list_from(grid=grid)

    x_caustic_radial, y_caustic_radial = (
        radial_caustic_list[0][:, 1],
        radial_caustic_list[0][:, 0],
    )

    assert np.mean(x_caustic_radial**2 + y_caustic_radial**2) == pytest.approx(
        mp.einstein_radius**2, 5e-1
    )

    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    radial_caustic_list = mp.radial_caustic_list_from(grid=grid)

    y_centre = np.mean(radial_caustic_list[0][:, 0])
    x_centre = np.mean(radial_caustic_list[0][:, 1])

    assert -0.2 < y_centre < 0.2
    assert -0.35 < x_centre < 0.35

    mp = ag.mp.IsothermalSph(centre=(0.5, 1.0), einstein_radius=2.0)

    radial_caustic_list = mp.radial_caustic_list_from(grid=grid)

    y_centre = np.mean(radial_caustic_list[0][:, 0])
    x_centre = np.mean(radial_caustic_list[0][:, 1])

    assert 0.3 < y_centre < 0.7
    assert 0.7 < x_centre < 1.2


def test__radial_caustic_list_from___compare_via_magnification():
    grid = ag.Grid2D.uniform(shape_native=(60, 60), pixel_scales=0.08)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=2, ell_comps=(0.109423, -0.019294)
    )

    caustic_radial_via_magnification = caustics_via_magnification_from(
        mass_profile=mp, grid=grid
    )[1]

    radial_caustic_list = mp.radial_caustic_list_from(grid=grid, pixel_scale=0.08)

    assert sum(radial_caustic_list[0]) == pytest.approx(
        sum(caustic_radial_via_magnification), 7e-1
    )


def test__radial_critical_curve_area_list_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    area_within_radial_critical_curve_list = mp.radial_critical_curve_area_list_from(
        grid=grid
    )

    assert area_within_radial_critical_curve_list[0] == pytest.approx(0.0455, 1e-1)


def test__tangential_critical_curve_area_list_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    area_calc = np.pi * mp.einstein_radius**2

    area_within_tangential_critical_curve_list = (
        mp.tangential_critical_curve_area_list_from(grid=grid)
    )

    assert area_within_tangential_critical_curve_list[0] == pytest.approx(
        area_calc, 1e-1
    )


def test__einstein_radius_list_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    einstein_radius_list = mp.einstein_radius_list_from(grid=grid)

    assert einstein_radius_list[0] == pytest.approx(2.0, 1e-1)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=2.0, ell_comps=(0.0, -0.25)
    )

    einstein_radius_list = mp.einstein_radius_list_from(grid=grid)

    assert einstein_radius_list[0] == pytest.approx(1.9360, 1e-1)


def test__einstein_radius_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    einstein_radius = mp.einstein_radius_from(grid=grid)

    assert einstein_radius == pytest.approx(2.0, 1e-1)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), einstein_radius=2.0, ell_comps=(0.0, -0.25)
    )

    einstein_radius = mp.einstein_radius_from(grid=grid)

    assert einstein_radius == pytest.approx(1.9360, 1e-1)


def test__einstein_mass_angular_list_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    einstein_mass_angular_list = mp.einstein_mass_angular_list_from(grid=grid)

    assert einstein_mass_angular_list[0] == pytest.approx(np.pi * 2.0**2.0, 1e-1)


def test__einstein_mass_angular_from():
    grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.2)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    einstein_mass_angular = mp.einstein_mass_angular_from(grid=grid)

    assert einstein_mass_angular == pytest.approx(np.pi * 2.0**2.0, 1e-1)


def test__jacobian_from():
    grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.05)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.0, -0.111111), einstein_radius=2.0
    )

    jacobian = mp.jacobian_from(grid=grid)

    A_12 = jacobian[0][1]
    A_21 = jacobian[1][0]

    mean_error = np.mean(A_12.slim - A_21.slim)

    assert mean_error < 1e-4


def test__convergence_2d_via_jacobian_from__compare_via_jacobian_and_analytic():
    grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05)

    mp = ag.mp.IsothermalSph(centre=(0.0, 0.0), einstein_radius=2.0)

    convergence_via_analytic = mp.convergence_2d_from(grid=grid)

    convergence_via_jacobian = mp.convergence_2d_via_jacobian_from(grid=grid)

    mean_error = np.mean(convergence_via_jacobian.slim - convergence_via_analytic.slim)

    assert convergence_via_jacobian.native.shape == (20, 20)
    assert mean_error < 1e-1

    mean_error = np.mean(convergence_via_jacobian.slim - convergence_via_analytic.slim)

    assert mean_error < 1e-1

    grid = ag.Grid2D.uniform(shape_native=(20, 20), pixel_scales=0.05)

    mp = ag.mp.Isothermal(
        centre=(0.0, 0.0), ell_comps=(0.111111, 0.0), einstein_radius=2.0
    )

    convergence_via_analytic = mp.convergence_2d_from(grid=grid)

    convergence_via_jacobian = mp.convergence_2d_via_jacobian_from(grid=grid)

    mean_error = np.mean(convergence_via_jacobian.slim - convergence_via_analytic.slim)

    assert mean_error < 1e-1


def test__evaluation_grid__changes_resolution_based_on_pixel_scale_input():
    from autogalaxy.operate.deflections import evaluation_grid

    @evaluation_grid
    def mock_func(lensing_obj, grid, pixel_scale=0.05):
        return grid

    grid = ag.Grid2D.uniform(shape_native=(4, 4), pixel_scales=0.05)

    evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=0.05)

    assert (evaluation_grid == grid).all()

    evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=0.1)
    downscaled_grid = ag.Grid2D.uniform(shape_native=(2, 2), pixel_scales=0.1)

    assert (evaluation_grid == downscaled_grid).all()

    evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=0.025)
    upscaled_grid = ag.Grid2D.uniform(shape_native=(8, 8), pixel_scales=0.025)

    assert (evaluation_grid == upscaled_grid).all()

    evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=0.03)
    upscaled_grid = ag.Grid2D.uniform(shape_native=(6, 6), pixel_scales=0.03)

    assert (evaluation_grid == upscaled_grid).all()


def test__evaluation_grid__changes_to_uniform_and_zoomed_in_if_masked():
    from autogalaxy.operate.deflections import evaluation_grid

    @evaluation_grid
    def mock_func(lensing_obj, grid, pixel_scale=0.05):
        return grid

    mask = ag.Mask2D.circular(shape_native=(11, 11), pixel_scales=1.0, radius=3.0)

    grid = ag.Grid2D.from_mask(mask=mask)

    evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=1.0)
    grid_uniform = ag.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

    assert (evaluation_grid[0] == np.array([3.0, -3.0])).all()
    assert (evaluation_grid == grid_uniform).all()

    mask = ag.Mask2D.circular(
        shape_native=(29, 29), pixel_scales=1.0, radius=3.0, centre=(5.0, 5.0)
    )

    grid = ag.Grid2D.from_mask(mask=mask)

    evaluation_grid = mock_func(lensing_obj=None, grid=grid, pixel_scale=1.0)
    grid_uniform = ag.Grid2D.uniform(
        shape_native=(7, 7), pixel_scales=1.0, origin=(5.0, 5.0)
    )

    assert (evaluation_grid[0] == np.array([8.0, 2.0])).all()
    assert (evaluation_grid == grid_uniform).all()
