import math
import numpy as np
import pytest

import autogalaxy as ag

from autogalaxy import exc


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


class Test1DFromGrid:
    def test__convergence_1d_from__grid_2d_in__returns_1d_image_via_projected_quantities(
        self
    ):

        grid_2d = ag.Grid2D.uniform(shape_native=(5, 5), pixel_scales=1.0)

        sie = ag.mp.EllIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), einstein_radius=1.0
        )

        convergence_1d = sie.convergence_1d_from(grid=grid_2d)
        convergence_2d = sie.convergence_2d_from(grid=grid_2d)

        assert convergence_1d[0] == pytest.approx(convergence_2d.native[2, 2], 1.0e-4)
        assert convergence_1d[1] == pytest.approx(convergence_2d.native[2, 3], 1.0e-4)
        assert convergence_1d[2] == pytest.approx(convergence_2d.native[2, 4], 1.0e-4)

        sie = ag.mp.EllIsothermal(
            centre=(0.2, 0.2), elliptical_comps=(0.3, 0.3), einstein_radius=1.0
        )

        convergence_1d = sie.convergence_1d_from(grid=grid_2d)

        grid_2d_projected = grid_2d.grid_2d_radial_projected_from(
            centre=sie.centre, angle=sie.angle + 90.0
        )

        convergence_projected = sie.convergence_2d_from(grid=grid_2d_projected)

        assert convergence_1d == pytest.approx(convergence_projected, 1.0e-4)
        assert (convergence_1d.grid_radial == np.array([0.0, 1.0, 2.0])).all()

    def test__convergence_1d_from__grid_2d_irregular_in__returns_1d_quantities(self):

        grid_2d = ag.Grid2DIrregular(grid=[[1.0, 1.0], [2.0, 2.0], [4.0, 4.0]])

        sie = ag.mp.EllIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), einstein_radius=1.0
        )

        convergence_1d = sie.convergence_1d_from(grid=grid_2d)
        convergence_2d = sie.convergence_2d_from(grid=grid_2d)

        assert convergence_1d[0] == pytest.approx(convergence_2d[0], 1.0e-4)
        assert convergence_1d[1] == pytest.approx(convergence_2d[1], 1.0e-4)
        assert convergence_1d[2] == pytest.approx(convergence_2d[2], 1.0e-4)

        sie = ag.mp.EllIsothermal(
            centre=(0.2, 0.2), elliptical_comps=(0.3, 0.3), einstein_radius=1.0
        )

        convergence_1d = sie.convergence_1d_from(grid=grid_2d)
        convergence_2d = sie.convergence_2d_from(grid=grid_2d)

        assert convergence_1d[0] == pytest.approx(convergence_2d[0], 1.0e-4)
        assert convergence_1d[1] == pytest.approx(convergence_2d[1], 1.0e-4)
        assert convergence_1d[2] == pytest.approx(convergence_2d[2], 1.0e-4)

    def test__convergence_1d_from__grid_1d_in__returns_1d_quantities_via_projection(
        self
    ):

        grid_1d = ag.Grid1D.manual_native(grid=[1.0, 2.0, 3.0], pixel_scales=1.0)

        sie = ag.mp.EllIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), einstein_radius=1.0
        )

        convergence_1d = sie.convergence_1d_from(grid=grid_1d)
        convergence_2d = sie.convergence_2d_from(grid=grid_1d)

        assert convergence_1d[0] == pytest.approx(convergence_2d[0], 1.0e-4)
        assert convergence_1d[1] == pytest.approx(convergence_2d[1], 1.0e-4)
        assert convergence_1d[2] == pytest.approx(convergence_2d[2], 1.0e-4)

        sie = ag.mp.EllIsothermal(
            centre=(0.5, 0.5), elliptical_comps=(0.2, 0.2), einstein_radius=1.0
        )

        convergence_1d = sie.convergence_1d_from(grid=grid_1d)

        grid_2d_radial = grid_1d.grid_2d_radial_projected_from(angle=sie.angle + 90.0)

        convergence_2d = sie.convergence_2d_from(grid=grid_2d_radial)

        assert convergence_1d[0] == pytest.approx(convergence_2d[0], 1.0e-4)
        assert convergence_1d[1] == pytest.approx(convergence_2d[1], 1.0e-4)
        assert convergence_1d[2] == pytest.approx(convergence_2d[2], 1.0e-4)

    def test__potential_1d_from__grid_2d_in__returns_1d_image_via_projected_quantities(
        self
    ):

        grid_2d = ag.Grid2D.uniform(shape_native=(5, 5), pixel_scales=1.0)

        sie = ag.mp.EllIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, 0.0), einstein_radius=1.0
        )

        potential_1d = sie.potential_1d_from(grid=grid_2d)
        potential_2d = sie.potential_2d_from(grid=grid_2d)

        assert potential_1d[0] == pytest.approx(potential_2d.native[2, 2], 1.0e-4)
        assert potential_1d[1] == pytest.approx(potential_2d.native[2, 3], 1.0e-4)
        assert potential_1d[2] == pytest.approx(potential_2d.native[2, 4], 1.0e-4)

        sie = ag.mp.EllIsothermal(
            centre=(0.2, 0.2), elliptical_comps=(0.3, 0.3), einstein_radius=1.0
        )

        potential_1d = sie.potential_1d_from(grid=grid_2d)

        grid_2d_projected = grid_2d.grid_2d_radial_projected_from(
            centre=sie.centre, angle=sie.angle + 90.0
        )

        potential_projected = sie.potential_2d_from(grid=grid_2d_projected)

        assert potential_1d == pytest.approx(potential_projected, 1.0e-4)
        assert (potential_1d.grid_radial == np.array([0.0, 1.0, 2.0])).all()


class TestDeflectionsViaPotential:
    def test__compare_sis_deflections_via_potential_and_calculation(self):
        sis = ag.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        grid = ag.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.05, sub_size=1)

        deflections_via_calculation = sis.deflections_yx_2d_from(grid=grid)

        deflections_via_potential = sis.deflections_2d_via_potential_2d_from(grid=grid)

        mean_error = np.mean(
            deflections_via_potential.slim - deflections_via_calculation.slim
        )

        assert mean_error < 1e-4

    def test__compare_sie_at_phi_45__deflections_via_potential_and_calculation(self):
        sie = ag.mp.EllIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.05, sub_size=1)

        deflections_via_calculation = sie.deflections_yx_2d_from(grid=grid)

        deflections_via_potential = sie.deflections_2d_via_potential_2d_from(grid=grid)

        mean_error = np.mean(
            deflections_via_potential.slim - deflections_via_calculation.slim
        )

        assert mean_error < 1e-4

    def test__compare_sie_at_phi_0__deflections_via_potential_and_calculation(self):
        sie = ag.mp.EllIsothermal(
            centre=(0.0, 0.0), elliptical_comps=(0.0, -0.111111), einstein_radius=2.0
        )

        grid = ag.Grid2D.uniform(shape_native=(10, 10), pixel_scales=0.05, sub_size=1)

        deflections_via_calculation = sie.deflections_yx_2d_from(grid=grid)

        deflections_via_potential = sie.deflections_2d_via_potential_2d_from(grid=grid)

        mean_error = np.mean(
            deflections_via_potential.slim - deflections_via_calculation.slim
        )

        assert mean_error < 1e-4


class TestMassWithin:
    def test__compare_to_analytic_and_grid_calculations(self):

        sis = ag.mp.SphIsothermal(einstein_radius=2.0)

        mass = sis.mass_angular_within_circle_from(radius=2.0)
        assert math.pi * sis.einstein_radius * 2.0 == pytest.approx(mass, 1e-3)

        sis = ag.mp.SphIsothermal(einstein_radius=4.0)

        mass = sis.mass_angular_within_circle_from(radius=4.0)
        assert math.pi * sis.einstein_radius * 4.0 == pytest.approx(mass, 1e-3)

        sis = ag.mp.SphIsothermal(einstein_radius=2.0)

        mass_grid = mass_within_radius_of_profile_from_grid_calculation(
            radius=1.0, profile=sis
        )

        mass = sis.mass_angular_within_circle_from(radius=1.0)

        assert mass_grid == pytest.approx(mass, 0.02)


class TestRadiusAverageConvergenceOne:
    def test__radius_of_average_convergence(self):

        sis = ag.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        assert sis.average_convergence_of_1_radius == pytest.approx(2.0, 1e-4)

        sie = ag.mp.EllIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0, elliptical_comps=(0.0, 0.111111)
        )

        assert sie.average_convergence_of_1_radius == pytest.approx(1.0, 1e-4)

        sie = ag.mp.EllIsothermal(
            centre=(0.0, 0.0), einstein_radius=3.0, elliptical_comps=(0.0, 0.333333)
        )

        assert sie.average_convergence_of_1_radius == pytest.approx(3.0, 1e-4)

        sie = ag.mp.EllIsothermal(
            centre=(0.0, 0.0), einstein_radius=8.0, elliptical_comps=(0.0, 0.666666)
        )

        assert sie.average_convergence_of_1_radius == pytest.approx(8.0, 1e-4)


class TestDensityBetweenAnnuli:
    def test__circular_annuli__sis__analyic_density_agrees(self):

        einstein_radius = 1.0

        sis = ag.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=einstein_radius)

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

        nfw = ag.mp.EllNFW(
            centre=(0.0, 0.0), elliptical_comps=(0.111111, 0.0), kappa_s=1.0
        )

        inner_mass = nfw.mass_angular_within_circle_from(radius=1.0)

        outer_mass = nfw.mass_angular_within_circle_from(radius=2.0)

        density_between_annuli = nfw.density_between_circular_annuli(
            inner_annuli_radius=1.0, outer_annuli_radius=2.0
        )

        annuli_area = (np.pi * 2.0 ** 2.0) - (np.pi * 1.0 ** 2.0)

        assert (outer_mass - inner_mass) / annuli_area == pytest.approx(
            density_between_annuli, 1e-4
        )


class TestNormalizationEinstienRadius:
    def test__mass_angular_via_normalization_from(self):

        sis = ag.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        mass_angular_from_normalization = sis.mass_angular_via_normalization_from(
            normalization=1.0, radius=2.0
        )

        assert mass_angular_from_normalization == pytest.approx(2.0 * np.pi, 1.0e-2)

        mass_angular_from_normalization = sis.mass_angular_via_normalization_from(
            normalization=1.0, radius=4.0
        )

        assert mass_angular_from_normalization == pytest.approx(4.0 * np.pi, 1.0e-2)

        nfw = ag.mp.SphNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        mass_angular_from_normalization = nfw.mass_angular_via_normalization_from(
            normalization=2.0, radius=2.0
        )

        assert mass_angular_from_normalization == pytest.approx(15.19525, 1.0e-4)

        sersic = ag.mp.SphSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=3.0,
            mass_to_light_ratio=1.0,
        )

        mass_angular_from_normalization = sersic.mass_angular_via_normalization_from(
            normalization=2.0, radius=2.0
        )

        sersic = ag.mp.SphSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=3.0,
            mass_to_light_ratio=2.0,
        )

        assert mass_angular_from_normalization == pytest.approx(28.32431, 1.0e-4)

        mass_angular_from_normalization = sersic.mass_angular_via_normalization_from(
            normalization=0.1, radius=2.0
        )

        assert mass_angular_from_normalization == pytest.approx(1.416215, 1.0e-2)

    def test__normalization_via_mass_angular_from(self):

        sersic = ag.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        normalization = sersic.normalization_via_mass_angular_from(
            mass_angular=5.0,
            radius=2.0,
            normalization_min=0.5,
            normalization_max=3.0,
            bins=5,
        )

        assert normalization == pytest.approx(0.79577, 1.0e-2)

        nfw = ag.mp.SphNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0)

        normalization = nfw.normalization_via_mass_angular_from(
            mass_angular=6.35829,
            radius=2.0,
            normalization_min=0.5,
            normalization_max=3.0,
            bins=5,
        )

        assert normalization == pytest.approx(0.83687, 1.0e-2)

        sersic = ag.mp.SphSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=3.0,
            mass_to_light_ratio=1.0,
        )

        normalization = sersic.normalization_via_mass_angular_from(
            mass_angular=2.15403,
            radius=2.0,
            normalization_min=0.01,
            normalization_max=30.0,
            bins=5,
        )

        sersic = sersic.with_new_normalization(normalization=normalization)

        assert normalization == pytest.approx(0.152097, 1.0e-2)

        with pytest.raises(exc.ProfileException):
            sersic.normalization_via_mass_angular_from(
                mass_angular=1.0,
                radius=2.0,
                normalization_min=1e-4,
                normalization_max=1e-3,
                bins=2,
            )

    def test__einstein_radius_via_normalization_from(self):

        sis = ag.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        einstein_radius_via_normalization_from = sis.einstein_radius_via_normalization_from(
            normalization=1.0
        )

        assert einstein_radius_via_normalization_from == pytest.approx(1.0, 1.0e-2)

        nfw = ag.mp.SphNFW(centre=(0.0, 0.0), kappa_s=1.0, scale_radius=1.0)

        einstein_radius_via_normalization_from = nfw.einstein_radius_via_normalization_from(
            normalization=2.0
        )

        assert einstein_radius_via_normalization_from == pytest.approx(2.35829, 1.0e-4)

        sersic = ag.mp.SphSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=3.0,
            mass_to_light_ratio=1.0,
        )

        einstein_radius_via_normalization_from = sersic.einstein_radius_via_normalization_from(
            normalization=1.0
        )
        einstein_radius_from_profile = sersic.average_convergence_of_1_radius

        assert einstein_radius_via_normalization_from == pytest.approx(
            einstein_radius_from_profile, 1.0e-4
        )

        einstein_radius_via_normalization_from = sersic.einstein_radius_via_normalization_from(
            normalization=0.1
        )

        assert einstein_radius_via_normalization_from == pytest.approx(0.381544, 1.0e-2)

        einstein_radius_via_normalization_from = sersic.einstein_radius_via_normalization_from(
            normalization=1e-4
        )

        assert einstein_radius_via_normalization_from == None

        einstein_radius_via_normalization_from = sersic.einstein_radius_via_normalization_from(
            normalization=1e9
        )

        assert einstein_radius_via_normalization_from == None

    def test__normalization_via_einstein_radius_from(self):

        sersic = ag.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        normalization = sersic.normalization_via_einstein_radius_from(
            einstein_radius=1.0, normalization_min=0.5, normalization_max=3.0, bins=5
        )

        assert normalization == pytest.approx(1.0, 1.0e-2)

        nfw = ag.mp.SphNFW(centre=(0.0, 0.0), kappa_s=3.0, scale_radius=1.0)

        normalization = nfw.normalization_via_einstein_radius_from(
            einstein_radius=2.35829,
            normalization_min=0.5,
            normalization_max=3.0,
            bins=5,
        )

        assert normalization == pytest.approx(2.0, 1.0e-2)

        sersic = ag.mp.SphSersic(
            centre=(0.0, 0.0),
            intensity=1.0,
            effective_radius=1.0,
            sersic_index=3.0,
            mass_to_light_ratio=1.0,
        )

        normalization = sersic.normalization_via_einstein_radius_from(
            einstein_radius=2.15403,
            normalization_min=0.01,
            normalization_max=30.0,
            bins=5,
        )

        assert normalization == pytest.approx(1.0, 1.0e-2)

        with pytest.raises(exc.ProfileException):
            sersic.normalization_via_einstein_radius_from(
                einstein_radius=1.0,
                normalization_min=1e-4,
                normalization_max=1e-3,
                bins=2,
            )


class TestExtractObject:
    def test__extract_works(self):

        sis = ag.mp.SphIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

        einstein_radii = sis.extract_attribute(
            cls=ag.mp.MassProfile, attr_name="einstein_radius"
        )

        assert einstein_radii.in_list[0] == 2.0

        centres = sis.extract_attribute(cls=ag.mp.MassProfile, attr_name="centre")

        assert centres.in_list[0] == (0.0, 0.0)

        assert (
            sis.extract_attribute(cls=ag.mp.MassProfile, attr_name="einstein_radiu")
            == None
        )
        sis.extract_attribute(cls=ag.lp.LightProfile, attr_name="einstein_radius")


class TestRegression:
    def test__centre_of_profile_in_right_place(self):

        grid = ag.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

        mass_profile = ag.mp.EllIsothermal(centre=(2.0, 1.0), einstein_radius=1.0)
        convergence = mass_profile.convergence_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = mass_profile.potential_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = mass_profile.deflections_yx_2d_from(grid=grid)
        assert deflections.native[1, 4, 0] > 0
        assert deflections.native[2, 4, 0] < 0
        assert deflections.native[1, 4, 1] > 0
        assert deflections.native[1, 3, 1] < 0

        mass_profile = ag.mp.SphIsothermal(centre=(2.0, 1.0), einstein_radius=1.0)
        convergence = mass_profile.convergence_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        mass_profile = ag.mp.SphIsothermal(centre=(2.0, 1.0), einstein_radius=1.0)
        potential = mass_profile.potential_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = mass_profile.deflections_yx_2d_from(grid=grid)
        assert deflections.native[1, 4, 0] > 0
        assert deflections.native[2, 4, 0] < 0
        assert deflections.native[1, 4, 1] > 0
        assert deflections.native[1, 3, 1] < 0

        grid = ag.Grid2DIterate.uniform(
            shape_native=(7, 7),
            pixel_scales=1.0,
            fractional_accuracy=0.99,
            sub_steps=[2, 4],
        )

        mass_profile = ag.mp.EllIsothermal(centre=(2.0, 1.0), einstein_radius=1.0)
        convergence = mass_profile.convergence_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = mass_profile.potential_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = mass_profile.deflections_yx_2d_from(grid=grid)
        assert deflections.native[1, 4, 0] >= 0
        assert deflections.native[2, 4, 0] <= 0
        assert deflections.native[1, 4, 1] >= 0
        assert deflections.native[1, 3, 1] <= 0

        mass_profile = ag.mp.SphIsothermal(centre=(2.0, 1.0), einstein_radius=1.0)

        convergence = mass_profile.convergence_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = mass_profile.potential_2d_from(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = mass_profile.deflections_yx_2d_from(grid=grid)
        assert deflections.native[1, 4, 0] >= 0
        assert deflections.native[2, 4, 0] <= 0
        assert deflections.native[1, 4, 1] >= 0
        assert deflections.native[1, 3, 1] <= 0


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

        grid = ag.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        mass_profile = ag.mp.EllIsothermal(centre=(0.08, 0.08), einstein_radius=1.0)

        deflections = mass_profile.deflections_yx_2d_from(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
        grid_sub_2 = ag.Grid2D.from_mask(mask=mask_sub_2)
        deflections_sub_2 = mass_profile.deflections_yx_2d_from(grid=grid_sub_2).binned

        assert deflections == pytest.approx(deflections_sub_2, 1.0e-6)

        grid = ag.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.99, sub_steps=[2, 4, 8]
        )

        mass_profile = ag.mp.EllIsothermal(centre=(0.08, 0.08), einstein_radius=1.0)

        deflections = mass_profile.deflections_yx_2d_from(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from(mask=mask, sub_size=4)
        grid_sub_4 = ag.Grid2D.from_mask(mask=mask_sub_4)
        deflections_sub_4 = mass_profile.deflections_yx_2d_from(grid=grid_sub_4).binned

        assert deflections[0, 0] == deflections_sub_4[0, 0]

        mask_sub_8 = mask.mask_new_sub_size_from(mask=mask, sub_size=8)
        grid_sub_8 = ag.Grid2D.from_mask(mask=mask_sub_8)
        deflections_sub_8 = mass_profile.deflections_yx_2d_from(grid=grid_sub_8).binned

        assert deflections[4, 0] == deflections_sub_8[4, 0]
