import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__deflections_via_integral_from():
    mp = ag.mp.SersicRadialGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    deflections = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(3.60324873535244, 1e-3)
    assert deflections[0, 1] == pytest.approx(2.3638898009652, 1e-3)

    mp = ag.mp.SersicRadialGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=-1.0,
    )

    deflections = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections[0, 0] == pytest.approx(0.97806399756448, 1e-3)
    assert deflections[0, 1] == pytest.approx(0.725459334118341, 1e-3)


def test__deflections_2d_via_mge_from():
    mp = ag.mp.SersicRadialGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=-1.0,
    )

    deflections_via_integral = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_mge = mp.deflections_2d_via_mge_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_mge, 1.0e-3)


def test__deflections_2d_via_cse_from():
    mp = ag.mp.SersicRadialGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    deflections_via_integral = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse, 1.0e-4)

    mp = ag.mp.SersicRadialGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=-1.0,
    )

    deflections_via_integral = mp.deflections_2d_via_integral_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )
    deflections_via_cse = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert deflections_via_integral == pytest.approx(deflections_via_cse, 1.0e-4)


def test__deflections_yx_2d_from():
    mp = ag.mp.SersicRadialGradient()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral, 1.0e-4)

    mp = ag.mp.SersicRadialGradientSph()

    deflections = mp.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflections_via_integral = mp.deflections_2d_via_cse_from(
        grid=ag.Grid2DIrregular([[1.0, 0.0]])
    )

    assert deflections == pytest.approx(deflections_via_integral, 1.0e-4)

    elliptical = ag.mp.SersicRadialGradient(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    spherical = ag.mp.SersicRadialGradient(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    ell_deflections_yx_2d = elliptical.deflections_yx_2d_from(grid=grid)
    sph_deflections_yx_2d = spherical.deflections_yx_2d_from(grid=grid)

    ell_deflections_yx_2d == pytest.approx(sph_deflections_yx_2d, 1.0e-4)


def test__convergence_2d_from():
    # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1/0.6)**-1.0 = 0.6
    mp = ag.mp.SersicRadialGradient(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[0.0, 1.0]]))

    assert convergence == pytest.approx(0.6 * 0.351797, 1e-3)

    # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = (1.5/2.0)**1.0 = 0.75

    mp = ag.mp.SersicRadialGradient(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=-1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.5, 0.0]]))

    assert convergence == pytest.approx(0.75 * 4.90657319276, 1e-3)

    mp = ag.mp.SersicRadialGradient(
        ell_comps=(0.0, 0.0),
        intensity=6.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=-1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.5, 0.0]]))

    assert convergence == pytest.approx(2.0 * 0.75 * 4.90657319276, 1e-3)

    mp = ag.mp.SersicRadialGradient(
        ell_comps=(0.0, 0.0),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=-1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.5, 0.0]]))

    assert convergence == pytest.approx(2.0 * 0.75 * 4.90657319276, 1e-3)

    # ((axis_ratio*radius/effective_radius)**-mass_to_light_gradient) = ((0.5*1.41)/2.0)**-1.0 = 2.836
    mp = ag.mp.SersicRadialGradient(
        ell_comps=(0.0, 0.333333),
        intensity=3.0,
        effective_radius=2.0,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    convergence = mp.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(2.836879 * 5.38066670129, abs=2e-01)

    elliptical = ag.mp.SersicRadialGradient(
        centre=(0.0, 0.0),
        ell_comps=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    spherical = ag.mp.SersicRadialGradient(
        centre=(0.0, 0.0),
        intensity=1.0,
        effective_radius=1.0,
        sersic_index=4.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=1.0,
    )

    ell_convergence_2d = elliptical.convergence_2d_from(grid=grid)
    sph_convergence_2d = spherical.convergence_2d_from(grid=grid)

    assert ell_convergence_2d == pytest.approx(sph_convergence_2d, 1.0e-4)


def test__compare_to_sersic():
    mp = ag.mp.SersicRadialGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=1.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=0.0,
    )

    sersic_deflections = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    exponential = ag.mp.Exponential(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        mass_to_light_ratio=1.0,
    )
    exponential_deflections = exponential.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert sersic_deflections[0, 0] == pytest.approx(
        exponential_deflections[0, 0], 1e-3
    )
    assert sersic_deflections[0, 0] == pytest.approx(0.90493, 1e-3)
    assert sersic_deflections[0, 1] == pytest.approx(
        exponential_deflections[0, 1], 1e-3
    )
    assert sersic_deflections[0, 1] == pytest.approx(0.62569, 1e-3)

    mp = ag.mp.SersicRadialGradient(
        centre=(0.4, 0.2),
        ell_comps=(0.0180010, 0.0494575),
        intensity=2.0,
        effective_radius=0.8,
        sersic_index=4.0,
        mass_to_light_ratio=3.0,
        mass_to_light_gradient=0.0,
    )
    sersic_deflections = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    dev = ag.mp.DevVaucouleurs(
        centre=(0.4, 0.2),
        ell_comps=(0.0180010, 0.0494575),
        intensity=2.0,
        effective_radius=0.8,
        mass_to_light_ratio=3.0,
    )

    dev_deflections = dev.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    # assert sersic_deflections[0, 0] == pytest.approx(dev_deflections[0, 0], 1e-3)
    # assert sersic_deflections[0, 0] == pytest.approx(-24.528, 1e-3)
    # assert sersic_deflections[0, 1] == pytest.approx(dev_deflections[0, 1], 1e-3)
    # assert sersic_deflections[0, 1] == pytest.approx(-3.37605, 1e-3)

    sersic_grad = ag.mp.SersicRadialGradient(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
        mass_to_light_gradient=0.0,
    )
    sersic_grad_deflections = sersic_grad.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    mp = ag.mp.Sersic(
        centre=(-0.4, -0.2),
        ell_comps=(-0.07142, -0.085116),
        intensity=5.0,
        effective_radius=0.2,
        sersic_index=2.0,
        mass_to_light_ratio=1.0,
    )
    sersic_deflections = mp.deflections_yx_2d_from(
        grid=ag.Grid2DIrregular([[0.1625, 0.1625]])
    )

    assert sersic_deflections[0, 0] == pytest.approx(
        sersic_grad_deflections[0, 0], 1e-3
    )
    assert sersic_deflections[0, 0] == pytest.approx(1.1446, 1e-3)
    assert sersic_deflections[0, 1] == pytest.approx(
        sersic_grad_deflections[0, 1], 1e-3
    )
    assert sersic_deflections[0, 1] == pytest.approx(0.79374, 1e-3)
