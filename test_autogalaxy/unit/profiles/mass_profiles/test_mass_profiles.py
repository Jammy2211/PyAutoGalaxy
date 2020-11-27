import math

import autogalaxy as ag
import numpy as np
import pytest


def mass_within_radius_of_profile_from_grid_calculation(radius, profile):

    mass_total = 0.0

    xs = np.linspace(-radius * 1.5, radius * 1.5, 40)
    ys = np.linspace(-radius * 1.5, radius * 1.5, 40)

    edge = xs[1] - xs[0]
    area = edge ** 2

    for x in xs:
        for y in ys:

            eta = profile.grid_to_elliptical_radii(grid=np.array([[x, y]]))

            if eta < radius:
                mass_total += profile.convergence_func(eta) * area

    return mass_total


class TestMassWithinCircle:
    def test__compare_to_analytic_and_grid_calculations(self):

        sis = ag.mp.SphericalIsothermal(einstein_radius=2.0)

        mass = sis.mass_angular_within_circle(radius=2.0)
        assert math.pi * sis.einstein_radius * 2.0 == pytest.approx(mass, 1e-3)

        sis = ag.mp.SphericalIsothermal(einstein_radius=4.0)

        mass = sis.mass_angular_within_circle(radius=4.0)
        assert math.pi * sis.einstein_radius * 4.0 == pytest.approx(mass, 1e-3)

        sis = ag.mp.SphericalIsothermal(einstein_radius=2.0)

        mass_grid = mass_within_radius_of_profile_from_grid_calculation(
            radius=1.0, profile=sis
        )

        mass = sis.mass_angular_within_circle(radius=1.0)

        assert mass_grid == pytest.approx(mass, 0.02)


class TestDensityBetweenAnnuli:
    def test__circular_annuli__sis__analyic_density_agrees(self):

        einstein_radius = 1.0

        sis = ag.mp.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=einstein_radius
        )

        inner_annuli_radius = 2.0
        outer_annuli_radius = 3.0

        inner_mass = math.pi * einstein_radius * inner_annuli_radius
        outer_mass = math.pi * einstein_radius * outer_annuli_radius

        density_between_annuli = sis.density_between_circular_annuli(
            inner_annuli_radius=inner_annuli_radius,
            outer_annuli_radius=outer_annuli_radius,
        )

        annuli_area = (np.pi * outer_annuli_radius ** 2.0) - (
            np.pi * inner_annuli_radius ** 2.0
        )

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(
            density_between_annuli, 1e-4
        )

    def test__circular_annuli__nfw_profile__compare_to_manual_mass(self):

        nfw = ag.mp.EllipticalNFW(
            centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), kappa_s=1.0
        )

        inner_mass = nfw.mass_angular_within_circle(radius=1.0)

        outer_mass = nfw.mass_angular_within_circle(radius=2.0)

        density_between_annuli = nfw.density_between_circular_annuli(
            inner_annuli_radius=1.0, outer_annuli_radius=2.0
        )

        annuli_area = (np.pi * 2.0 ** 2.0) - (np.pi * 1.0 ** 2.0)

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(
            density_between_annuli, 1e-4
        )


class TestLensingObject:
    def test__mass_profiles__list__is_list_of_self(self):

        sis = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        assert sis.mass_profiles == [sis]

    def test__correct_einstein_mass_caclulated__means_all_innherited_methods_work(
        self,
    ):

        sis = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        assert sis.einstein_mass_angular_via_tangential_critical_curve == pytest.approx(
            np.pi * 2.0 ** 2.0, 1.0e-1
        )


class TestRegression:
    def test__centre_of_profile_in_right_place(self):

        grid = ag.Grid.uniform(shape_2d=(7, 7), pixel_scales=1.0)

        mass_profile = ag.mp.EllipticalIsothermal(
            centre=(2.0, 1.0), einstein_radius=1.0
        )
        convergence = mass_profile.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = mass_profile.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = mass_profile.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] > 0
        assert deflections.in_2d[2, 4, 0] < 0
        assert deflections.in_2d[1, 4, 1] > 0
        assert deflections.in_2d[1, 3, 1] < 0

        mass_profile = ag.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0)
        convergence = mass_profile.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        mass_profile = ag.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0)
        potential = mass_profile.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = mass_profile.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] > 0
        assert deflections.in_2d[2, 4, 0] < 0
        assert deflections.in_2d[1, 4, 1] > 0
        assert deflections.in_2d[1, 3, 1] < 0

        grid = ag.GridIterate.uniform(
            shape_2d=(7, 7),
            pixel_scales=1.0,
            fractional_accuracy=0.99,
            sub_steps=[2, 4],
        )

        mass_profile = ag.mp.EllipticalIsothermal(
            centre=(2.0, 1.0), einstein_radius=1.0
        )
        convergence = mass_profile.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = mass_profile.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = mass_profile.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] >= 0
        assert deflections.in_2d[2, 4, 0] <= 0
        assert deflections.in_2d[1, 4, 1] >= 0
        assert deflections.in_2d[1, 3, 1] <= 0

        mass_profile = ag.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0)

        convergence = mass_profile.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = mass_profile.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = mass_profile.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] >= 0
        assert deflections.in_2d[2, 4, 0] <= 0
        assert deflections.in_2d[1, 4, 1] >= 0
        assert deflections.in_2d[1, 3, 1] <= 0


class TestDecorators:
    def test__grid_iterate_in__iterates_grid_result_correctly(self, gal_x1_mp):

        mask = ag.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = ag.GridIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        mass_profile = ag.mp.EllipticalIsothermal(
            centre=(0.08, 0.08), einstein_radius=1.0
        )

        deflections = mass_profile.deflections_from_grid(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = ag.Grid.from_mask(mask=mask_sub_2)
        deflections_sub_2 = mass_profile.deflections_from_grid(
            grid=grid_sub_2
        ).in_1d_binned

        assert deflections == pytest.approx(deflections_sub_2, 1.0e-6)

        grid = ag.GridIterate.from_mask(
            mask=mask, fractional_accuracy=0.99, sub_steps=[2, 4, 8]
        )

        mass_profile = ag.mp.EllipticalIsothermal(
            centre=(0.08, 0.08), einstein_radius=1.0
        )

        deflections = mass_profile.deflections_from_grid(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = ag.Grid.from_mask(mask=mask_sub_4)
        deflections_sub_4 = mass_profile.deflections_from_grid(
            grid=grid_sub_4
        ).in_1d_binned

        assert deflections[0, 0] == deflections_sub_4[0, 0]

        mask_sub_8 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)
        grid_sub_8 = ag.Grid.from_mask(mask=mask_sub_8)
        deflections_sub_8 = mass_profile.deflections_from_grid(
            grid=grid_sub_8
        ).in_1d_binned

        assert deflections[4, 0] == deflections_sub_8[4, 0]

    def test__grid_interpolate_in__convergence__interpolates_based_on_intepolate_config(
        self,
    ):

        # `False` in interpolate.ini

        mask = ag.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = ag.Grid.from_mask(mask=mask)

        grid_interpolate = ag.GridInterpolate.from_mask(
            mask=mask, pixel_scales_interp=0.1
        )

        mass_profile = ag.mp.EllipticalIsothermal(einstein_radius=1.0)

        convergence = mass_profile.convergence_from_grid(grid=grid)
        convergence_no_interpolate = mass_profile.convergence_from_grid(
            grid=grid_interpolate
        )

        assert (convergence == convergence_no_interpolate).all()

        # `False` in interpolate.ini

        mass_profile = ag.mp.SphericalIsothermal(einstein_radius=1.0)

        convergence = mass_profile.convergence_from_grid(grid=grid)
        convergence_interpolate = mass_profile.convergence_from_grid(
            grid=grid_interpolate
        )
        assert (convergence != convergence_interpolate).all()

        array_interp = mass_profile.convergence_from_grid(
            grid=grid_interpolate.grid_interp
        )
        interpolated_array = grid_interpolate.interpolated_array_from_array_interp(
            array_interp=array_interp
        )
        assert (convergence_interpolate == interpolated_array).all()

    def test__grid_interpolate_in__potential__interpolates_based_on_intepolate_config(
        self,
    ):

        # `False` in interpolate.ini

        mask = ag.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = ag.Grid.from_mask(mask=mask)

        grid_interpolate = ag.GridInterpolate.from_mask(
            mask=mask, pixel_scales_interp=0.1
        )

        mass_profile = ag.mp.EllipticalIsothermal(einstein_radius=1.0)

        potential = mass_profile.potential_from_grid(grid=grid)
        potential_no_interpolate = mass_profile.potential_from_grid(
            grid=grid_interpolate
        )

        assert (potential == potential_no_interpolate).all()

        # `False` in interpolate.ini

        mass_profile = ag.mp.SphericalIsothermal(einstein_radius=1.0)

        potential = mass_profile.potential_from_grid(grid=grid)
        potential_interpolate = mass_profile.potential_from_grid(grid=grid_interpolate)
        assert (potential != potential_interpolate).all()

        array_interp = mass_profile.potential_from_grid(
            grid=grid_interpolate.grid_interp
        )
        interpolated_array = grid_interpolate.interpolated_array_from_array_interp(
            array_interp=array_interp
        )
        assert (potential_interpolate == interpolated_array).all()

    def test__grid_interpolate_in__deflections__interpolates_based_on_intepolate_config(
        self,
    ):

        # `False` in interpolate.ini

        mask = ag.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
        )

        grid = ag.Grid.from_mask(mask=mask)

        grid_interpolate = ag.GridInterpolate.from_mask(
            mask=mask, pixel_scales_interp=0.1
        )

        mass_profile = ag.mp.EllipticalIsothermal(einstein_radius=1.0)

        deflections = mass_profile.deflections_from_grid(grid=grid)
        deflections_no_interpolate = mass_profile.deflections_from_grid(
            grid=grid_interpolate
        )

        assert (deflections == deflections_no_interpolate).all()

        # `False` in interpolate.ini

        mass_profile = ag.mp.SphericalIsothermal(einstein_radius=1.0)

        deflections_interpolate = mass_profile.deflections_from_grid(
            grid=grid_interpolate
        )

        grid_interp = mass_profile.deflections_from_grid(
            grid=grid_interpolate.grid_interp
        )
        interpolated_grid = grid_interpolate.interpolated_grid_from_grid_interp(
            grid_interp=grid_interp
        )
        assert (deflections_interpolate == interpolated_grid).all()
