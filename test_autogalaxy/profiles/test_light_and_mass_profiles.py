import autogalaxy as ag
import numpy as np
import pytest

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__gaussian():
    gaussian_lp = ag.lmp.Gaussian(ell_comps=(0.1, 0.05), intensity=1.0, sigma=5.0)
    gaussian_mp = ag.lmp.Gaussian(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )
    gaussian_lmp = ag.lmp.Gaussian(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        sigma=5.0,
        mass_to_light_ratio=2.0,
    )

    assert (
        gaussian_lp.image_2d_from(grid=grid) == gaussian_lmp.image_2d_from(grid=grid)
    ).all()
    assert (
        gaussian_mp.convergence_2d_from(grid=grid)
        == gaussian_lmp.convergence_2d_from(grid=grid)
    ).all()
    #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
    assert (
        gaussian_mp.deflections_yx_2d_from(grid=grid)
        == gaussian_lmp.deflections_yx_2d_from(grid=grid)
    ).all()


def test__sersic():
    sersic_lp = ag.lmp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
    )
    sersic_mp = ag.lmp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )
    sersic_lmp = ag.lmp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    assert (
        sersic_lp.image_2d_from(grid=grid) == sersic_lmp.image_2d_from(grid=grid)
    ).all()
    assert (
        sersic_mp.convergence_2d_from(grid=grid)
        == sersic_lmp.convergence_2d_from(grid=grid)
    ).all()
    #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
    assert (
        sersic_mp.deflections_yx_2d_from(grid=grid)
        == sersic_lmp.deflections_yx_2d_from(grid=grid)
    ).all()


def test__exponential():
    sersic_lp = ag.lmp.Exponential(
        ell_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6
    )
    sersic_mp = ag.lmp.Exponential(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )
    sersic_lmp = ag.lmp.Exponential(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )

    assert (
        sersic_lp.image_2d_from(grid=grid) == sersic_lmp.image_2d_from(grid=grid)
    ).all()
    assert (
        sersic_mp.convergence_2d_from(grid=grid)
        == sersic_lmp.convergence_2d_from(grid=grid)
    ).all()
    #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
    assert (
        sersic_mp.deflections_yx_2d_from(grid=grid)
        == sersic_lmp.deflections_yx_2d_from(grid=grid)
    ).all()


def test__dev_vaucouleurs():
    sersic_lp = ag.lmp.DevVaucouleurs(
        ell_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6
    )
    sersic_mp = ag.lmp.DevVaucouleurs(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )
    sersic_lmp = ag.lmp.DevVaucouleurs(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
    )

    assert (
        sersic_lp.image_2d_from(grid=grid) == sersic_lmp.image_2d_from(grid=grid)
    ).all()
    assert (
        sersic_mp.convergence_2d_from(grid=grid)
        == sersic_lmp.convergence_2d_from(grid=grid)
    ).all()
    #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
    assert (
        sersic_mp.deflections_yx_2d_from(grid=grid)
        == sersic_lmp.deflections_yx_2d_from(grid=grid)
    ).all()


def test__sersic_radial_gradient():
    sersic_lp = ag.lmp.Sersic(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
    )
    sersic_mp = ag.lmp.SersicRadialGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )
    sersic_lmp = ag.lmp.SersicRadialGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )

    assert sersic_lp.image_2d_from(grid=grid) == pytest.approx(
        sersic_lmp.image_2d_from(grid=grid), 1.0e-4
    )
    assert sersic_mp.convergence_2d_from(grid=grid) == pytest.approx(
        sersic_lmp.convergence_2d_from(grid=grid), 1.0e-4
    )
    #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
    assert sersic_mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
        sersic_lmp.deflections_yx_2d_from(grid=grid), 1.0e-4
    )


def test__sersic_radial_gradient():
    sersic_lp = ag.lmp.Exponential(
        ell_comps=(0.1, 0.05), intensity=1.0, effective_radius=0.6
    )
    sersic_mp = ag.lmp.ExponentialRadialGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )
    sersic_lmp = ag.lmp.ExponentialRadialGradient(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        effective_radius=0.6,
        mass_to_light_ratio=2.0,
        mass_to_light_gradient=0.5,
    )

    assert sersic_lp.image_2d_from(grid=grid) == pytest.approx(
        sersic_lmp.image_2d_from(grid=grid), 1.0e-4
    )
    assert sersic_mp.convergence_2d_from(grid=grid) == pytest.approx(
        sersic_lmp.convergence_2d_from(grid=grid), 1.0e-4
    )
    #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
    assert sersic_mp.deflections_yx_2d_from(grid=grid) == pytest.approx(
        sersic_lmp.deflections_yx_2d_from(grid=grid), 1.0e-4
    )


def test__core_sersic():
    sersic_lp = ag.lmp.SersicCore(
        ell_comps=(0.1, 0.05), effective_radius=0.6, sersic_index=2.0
    )
    sersic_mp = ag.lmp.SersicCore(
        ell_comps=(0.1, 0.05),
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )
    sersic_lmp = ag.lmp.SersicCore(
        ell_comps=(0.1, 0.05),
        effective_radius=0.6,
        sersic_index=2.0,
        mass_to_light_ratio=2.0,
    )

    assert (
        sersic_lp.image_2d_from(grid=grid) == sersic_lmp.image_2d_from(grid=grid)
    ).all()
    assert (
        sersic_mp.convergence_2d_from(grid=grid)
        == sersic_lmp.convergence_2d_from(grid=grid)
    ).all()
    #    assert (sersic_mp.potential_2d_from(grid=grid) == sersic_lmp.potential_2d_from(grid=grid)).all()
    assert (
        sersic_mp.deflections_yx_2d_from(grid=grid)
        == sersic_lmp.deflections_yx_2d_from(grid=grid)
    ).all()


def test__chameleon():
    chameleon_lp = ag.lmp.Chameleon(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
    )
    chameleon_mp = ag.lmp.Chameleon(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
        mass_to_light_ratio=2.0,
    )
    chameleon_lmp = ag.lmp.Chameleon(
        ell_comps=(0.1, 0.05),
        intensity=1.0,
        core_radius_0=0.1,
        core_radius_1=0.3,
        mass_to_light_ratio=2.0,
    )

    assert (
        chameleon_lp.image_2d_from(grid=grid) == chameleon_lmp.image_2d_from(grid=grid)
    ).all()
    assert (
        chameleon_mp.convergence_2d_from(grid=grid)
        == chameleon_lmp.convergence_2d_from(grid=grid)
    ).all()
    #    assert (chameleon_mp.potential_2d_from(grid=grid) == chameleon_lmp.potential_2d_from(grid=grid)).all()
    assert (
        chameleon_mp.deflections_yx_2d_from(grid=grid)
        == chameleon_lmp.deflections_yx_2d_from(grid=grid)
    ).all()
