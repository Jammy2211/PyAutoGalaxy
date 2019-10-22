import numpy as np
import pytest
from skimage import measure

import autoarray as aa
import autoastro as am
from autoastro import exc
from test_autoastro.mock import mock_cosmology


class TestLightProfiles(object):
    class TestProfileImage:
        def test__no_light_profiles__profile_image_returned_as_0s_of_shape_grid(
            self, sub_grid_7x7
        ):
            galaxy = am.Galaxy(redshift=0.5)

            profile_image = galaxy.profile_image_from_grid(grid=sub_grid_7x7)

            assert (profile_image == np.zeros(shape=sub_grid_7x7.shape[0])).all()

        def test__using_no_light_profiles__check_reshaping_decorator_of_returned_profile_image(
            self, sub_grid_7x7
        ):
            galaxy = am.Galaxy(redshift=0.5)

            profile_image = galaxy.profile_image_from_grid(grid=sub_grid_7x7)

            assert (profile_image.in_2d_binned == np.zeros(shape=(7, 7))).all()

            profile_image = galaxy.profile_image_from_grid(grid=sub_grid_7x7)

            assert (profile_image == np.zeros(shape=sub_grid_7x7.shape[0])).all()

            profile_image = galaxy.profile_image_from_grid(grid=sub_grid_7x7)

            assert (
                profile_image.in_1d_binned == np.zeros(shape=sub_grid_7x7.shape[0] // 4)
            ).all()

        def test__galaxies_with_x1_and_x2_light_profiles__profile_image_is_same_individual_profiles(
            self, lp_0, gal_x1_lp, lp_1, gal_x2_lp
        ):
            lp_profile_image = lp_0.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )

            gal_lp_profile_image = gal_x1_lp.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )

            assert lp_profile_image == gal_lp_profile_image

            lp_profile_image = lp_0.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )
            lp_profile_image += lp_1.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )

            gal_profile_image = gal_x2_lp.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )

            assert lp_profile_image == gal_profile_image

        def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper_by_binning_sum_of_light_profile_values(
            self, sub_grid_7x7, gal_x2_lp
        ):

            lp_0_profile_image = gal_x2_lp.light_profile_0.profile_image_from_grid(
                grid=sub_grid_7x7
            )

            lp_1_profile_image = gal_x2_lp.light_profile_1.profile_image_from_grid(
                grid=sub_grid_7x7
            )

            lp_profile_image = lp_0_profile_image + lp_1_profile_image

            lp_profile_image_0 = (
                lp_profile_image[0]
                + lp_profile_image[1]
                + lp_profile_image[2]
                + lp_profile_image[3]
            ) / 4.0

            lp_profile_image_1 = (
                lp_profile_image[4]
                + lp_profile_image[5]
                + lp_profile_image[6]
                + lp_profile_image[7]
            ) / 4.0

            gal_profile_image = gal_x2_lp.profile_image_from_grid(grid=sub_grid_7x7)

            assert gal_profile_image.in_1d_binned[0] == lp_profile_image_0
            assert gal_profile_image.in_1d_binned[1] == lp_profile_image_1

    class TestLuminosityWithin:
        def test__two_profile_galaxy__is_sum_of_individual_profiles(
            self, lp_0, lp_1, gal_x1_lp, gal_x2_lp
        ):
            radius = am.dim.Length(0.5, "arcsec")

            lp_luminosity = lp_0.luminosity_within_circle_in_units(
                radius=radius, unit_luminosity="eps"
            )
            gal_luminosity = gal_x1_lp.luminosity_within_circle_in_units(
                radius=radius, unit_luminosity="eps"
            )

            assert lp_luminosity == gal_luminosity

            lp_luminosity = lp_0.luminosity_within_ellipse_in_units(
                major_axis=radius, unit_luminosity="eps"
            )
            lp_luminosity += lp_1.luminosity_within_ellipse_in_units(
                major_axis=radius, unit_luminosity="eps"
            )

            gal_luminosity = gal_x2_lp.luminosity_within_ellipse_in_units(
                major_axis=radius, unit_luminosity="eps"
            )

            assert lp_luminosity == gal_luminosity

        def test__radius_unit_conversions__multiply_by_kpc_per_arcsec(
            self, lp_0, gal_x1_lp
        ):
            cosmology = mock_cosmology.MockCosmology(
                arcsec_per_kpc=0.5, kpc_per_arcsec=2.0
            )

            radius = am.dim.Length(0.5, "arcsec")

            lp_luminosity_arcsec = lp_0.luminosity_within_circle_in_units(radius=radius)
            gal_luminosity_arcsec = gal_x1_lp.luminosity_within_circle_in_units(
                radius=radius
            )

            assert lp_luminosity_arcsec == gal_luminosity_arcsec

            radius = am.dim.Length(0.5, "kpc")

            lp_luminosity_kpc = lp_0.luminosity_within_circle_in_units(
                radius=radius, redshift_profile=0.5, cosmology=cosmology
            )
            gal_luminosity_kpc = gal_x1_lp.luminosity_within_circle_in_units(
                radius=radius, cosmology=cosmology
            )

            assert lp_luminosity_kpc == gal_luminosity_kpc

        def test__luminosity_unit_conversions__multiply_by_exposure_time(
            self, lp_0, gal_x1_lp
        ):
            radius = am.dim.Length(0.5, "arcsec")

            lp_luminosity_eps = lp_0.luminosity_within_ellipse_in_units(
                major_axis=radius, unit_luminosity="eps", exposure_time=2.0
            )
            gal_luminosity_eps = gal_x1_lp.luminosity_within_ellipse_in_units(
                major_axis=radius, unit_luminosity="eps", exposure_time=2.0
            )

            assert lp_luminosity_eps == gal_luminosity_eps

            lp_luminosity_counts = lp_0.luminosity_within_circle_in_units(
                radius=radius, unit_luminosity="counts", exposure_time=2.0
            )

            gal_luminosity_counts = gal_x1_lp.luminosity_within_circle_in_units(
                radius=radius, unit_luminosity="counts", exposure_time=2.0
            )

            assert lp_luminosity_counts == gal_luminosity_counts

        def test__no_light_profile__returns_none(self):
            gal_no_lp = am.Galaxy(
                redshift=0.5, mass=am.mp.SphericalIsothermal()
            )

            assert gal_no_lp.luminosity_within_circle_in_units(radius=1.0) == None
            assert gal_no_lp.luminosity_within_ellipse_in_units(major_axis=1.0) == None

    class TestSymmetricProfiles(object):
        def test_1d_symmetry(self):
            lp_0 = am.lp.EllipticalSersic(
                axis_ratio=1.0,
                phi=0.0,
                intensity=1.0,
                effective_radius=0.6,
                sersic_index=4.0,
            )

            lp_1 = am.lp.EllipticalSersic(
                axis_ratio=1.0,
                phi=0.0,
                intensity=1.0,
                effective_radius=0.6,
                sersic_index=4.0,
                centre=(100, 0),
            )

            gal_x2_lp = am.Galaxy(
                redshift=0.5, light_profile_0=lp_0, light_profile_1=lp_1
            )

            assert gal_x2_lp.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[0.0, 0.0]]])
            ) == gal_x2_lp.profile_image_from_grid(grid=aa.grid.manual_2d([[[100.0, 0.0]]]))
            assert gal_x2_lp.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 0.0]]])
            ) == gal_x2_lp.profile_image_from_grid(grid=aa.grid.manual_2d([[[51.0, 0.0]]]))

        def test_2d_symmetry(self):
            lp_0 = am.lp.EllipticalSersic(
                axis_ratio=1.0,
                phi=0.0,
                intensity=1.0,
                effective_radius=0.6,
                sersic_index=4.0,
            )

            lp_1 = am.lp.EllipticalSersic(
                axis_ratio=1.0,
                phi=0.0,
                intensity=1.0,
                effective_radius=0.6,
                sersic_index=4.0,
                centre=(100, 0),
            )

            lp_2 = am.lp.EllipticalSersic(
                axis_ratio=1.0,
                phi=0.0,
                intensity=1.0,
                effective_radius=0.6,
                sersic_index=4.0,
                centre=(0, 100),
            )

            lp_3 = am.lp.EllipticalSersic(
                axis_ratio=1.0,
                phi=0.0,
                intensity=1.0,
                effective_radius=0.6,
                sersic_index=4.0,
                centre=(100, 100),
            )

            gal_x4_lp = am.Galaxy(
                redshift=0.5,
                light_profile_0=lp_0,
                light_profile_1=lp_1,
                light_profile_3=lp_2,
                light_profile_4=lp_3,
            )

            assert gal_x4_lp.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 0.0]]])
            ) == pytest.approx(
                gal_x4_lp.profile_image_from_grid(grid=aa.grid.manual_2d([[[51.0, 0.0]]])), 1e-5
            )

            assert gal_x4_lp.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[0.0, 49.0]]])
            ) == pytest.approx(
                gal_x4_lp.profile_image_from_grid(grid=aa.grid.manual_2d([[[0.0, 51.0]]])), 1e-5
            )

            assert gal_x4_lp.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[100.0, 49.0]]])
            ) == pytest.approx(
                gal_x4_lp.profile_image_from_grid(grid=aa.grid.manual_2d([[[100.0, 51.0]]])), 1e-5
            )

            assert gal_x4_lp.profile_image_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 49.0]]])
            ) == pytest.approx(
                gal_x4_lp.profile_image_from_grid(grid=aa.grid.manual_2d([[[51.0, 51.0]]])), 1e-5
            )

    class TestBlurredProfileImages(object):
        def test__blurred_image_from_grid_and_psf(
            self, sub_grid_7x7, blurring_grid_7x7, psf_3x3, convolver_7x7
        ):
            light_profile_0 = am.lp.EllipticalSersic(intensity=2.0)
            light_profile_1 = am.lp.EllipticalSersic(intensity=3.0)

            galaxy = am.Galaxy(
                light_profile_0=light_profile_0,
                light_profile_1=light_profile_1,
                redshift=0.5,
            )

            image = galaxy.profile_image_from_grid(grid=sub_grid_7x7)

            blurring_image = galaxy.profile_image_from_grid(grid=blurring_grid_7x7)

            blurred_image = convolver_7x7.convolved_image_1d_from_image_array_and_blurring_array(
                image_array=image.in_1d_binned,
                blurring_array=blurring_image.in_1d_binned,
            )

            light_profile_blurred_image = galaxy.blurred_profile_image_from_grid_and_psf(
                grid=sub_grid_7x7, blurring_grid=blurring_grid_7x7, psf=psf_3x3
            )

            assert blurred_image.in_1d == pytest.approx(
                light_profile_blurred_image.in_1d, 1.0e-4
            )

            assert blurred_image.in_2d == pytest.approx(
                light_profile_blurred_image.in_2d, 1.0e-4
            )

        def test__blurred_image_from_grid_and_convolver(
            self, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
        ):
            light_profile_0 = am.lp.EllipticalSersic(intensity=2.0)
            light_profile_1 = am.lp.EllipticalSersic(intensity=3.0)

            galaxy = am.Galaxy(
                light_profile_0=light_profile_0,
                light_profile_1=light_profile_1,
                redshift=0.5,
            )

            image = galaxy.profile_image_from_grid(grid=sub_grid_7x7)

            blurring_image = galaxy.profile_image_from_grid(grid=blurring_grid_7x7)

            blurred_image = convolver_7x7.convolved_image_1d_from_image_array_and_blurring_array(
                image_array=image.in_1d_binned,
                blurring_array=blurring_image.in_1d_binned,
            )

            light_profile_blurred_image = galaxy.blurred_profile_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            assert blurred_image.in_1d == pytest.approx(
                light_profile_blurred_image.in_1d, 1.0e-4
            )

            assert blurred_image.in_2d == pytest.approx(
                light_profile_blurred_image.in_2d, 1.0e-4
            )

    class TestVisibilities(object):
        def test__visibilities_from_grid_and_transformer(
            self, sub_grid_7x7, transformer_7x7_7
        ):

            light_profile_0 = am.lp.EllipticalSersic(intensity=2.0)
            light_profile_1 = am.lp.EllipticalSersic(intensity=3.0)

            image = light_profile_0.profile_image_from_grid(
                grid=sub_grid_7x7
            ).in_1d_binned + light_profile_1.profile_image_from_grid(grid=sub_grid_7x7).in_1d_binned

            visibilities = transformer_7x7_7.visibilities_from_image(
                image=image
            )

            galaxy = am.Galaxy(
                light_profile_0=light_profile_0,
                light_profile_1=light_profile_1,
                redshift=0.5,
            )

            galaxy_visibilities = galaxy.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            assert visibilities == pytest.approx(galaxy_visibilities, 1.0e-4)


def critical_curve_via_magnification_from_galaxy_and_grid(galaxy, grid):
    magnification = galaxy.magnification_from_grid(grid=grid)

    inverse_magnification = 1 / magnification

    critical_curves_indices = measure.find_contours(inverse_magnification.in_2d, 0)

    no_critical_curves = len(critical_curves_indices)
    contours = []
    critical_curves = []

    for jj in np.arange(no_critical_curves):
        contours.append(critical_curves_indices[jj])
        contour_x, contour_y = contours[jj].T
        pixel_coord = np.stack((contour_x, contour_y), axis=-1)

        critical_curve = grid.geometry.grid_arcsec_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=pixel_coord, shape_2d=magnification.sub_shape_2d
        )

        critical_curves.append(critical_curve)

    return critical_curves


def caustics_via_magnification_from_galaxy_and_grid(galaxy, grid):
    caustics = []

    critical_curves = critical_curve_via_magnification_from_galaxy_and_grid(
        galaxy=galaxy, grid=grid
    )

    for i in range(len(critical_curves)):
        critical_curve = critical_curves[i]

        deflections = galaxy.deflections_from_grid(grid=critical_curve)

        caustic = critical_curve - deflections

        caustics.append(caustic)

    return caustics


class TestMassProfiles(object):
    class TestConvergence:
        def test__no_mass_profiles__convergence_returned_as_0s_of_shape_grid(
            self, sub_grid_7x7
        ):
            galaxy = am.Galaxy(redshift=0.5)

            convergence = galaxy.convergence_from_grid(grid=sub_grid_7x7)

            assert (convergence.in_1d == np.zeros(shape=sub_grid_7x7.shape[0])).all()

        def test__using_no_mass_profiles__check_reshaping_decorator_of_returned_convergence(
            self, sub_grid_7x7
        ):
            galaxy = am.Galaxy(redshift=0.5)

            convergence = galaxy.convergence_from_grid(grid=sub_grid_7x7)

            assert (convergence.in_2d_binned == np.zeros(shape=(7, 7))).all()

            convergence = galaxy.convergence_from_grid(grid=sub_grid_7x7)

            assert (convergence.in_1d == np.zeros(shape=sub_grid_7x7.shape[0])).all()

            convergence = galaxy.convergence_from_grid(grid=sub_grid_7x7)

            assert (convergence.in_1d_binned == np.zeros(shape=sub_grid_7x7.shape[0] // 4)).all()

        def test__galaxies_with_x1_and_x2_mass_profiles__convergence_is_same_individual_profiles(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):
            mp_convergence = mp_0.convergence_from_grid(grid=aa.grid.manual_2d([[[1.05, -0.55]]]))

            gal_mp_convergence = gal_x1_mp.convergence_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )

            assert mp_convergence == gal_mp_convergence

            mp_convergence = mp_0.convergence_from_grid(grid=aa.grid.manual_2d([[[1.05, -0.55]]]))
            mp_convergence += mp_1.convergence_from_grid(grid=aa.grid.manual_2d([[[1.05, -0.55]]]))

            gal_convergence = gal_x2_mp.convergence_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )

            assert mp_convergence == gal_convergence

        def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper_by_binning_sum_of_mass_profile_values(
            self, sub_grid_7x7, gal_x2_mp
        ):
            mp_0_convergence = gal_x2_mp.mass_profile_0.convergence_from_grid(
                grid=sub_grid_7x7
            )

            mp_1_convergence = gal_x2_mp.mass_profile_1.convergence_from_grid(
                grid=sub_grid_7x7
            )

            mp_convergence = mp_0_convergence + mp_1_convergence

            mp_convergence_0 = (
                mp_convergence[0]
                + mp_convergence[1]
                + mp_convergence[2]
                + mp_convergence[3]
            ) / 4.0

            mp_convergence_1 = (
                mp_convergence[4]
                + mp_convergence[5]
                + mp_convergence[6]
                + mp_convergence[7]
            ) / 4.0

            gal_convergence = gal_x2_mp.convergence_from_grid(grid=sub_grid_7x7)

            assert gal_convergence.in_1d_binned[0] == mp_convergence_0
            assert gal_convergence.in_1d_binned[1] == mp_convergence_1

    class TestPotential:
        def test__no_mass_profiles__potential_returned_as_0s_of_shape_grid(
            self, sub_grid_7x7
        ):
            galaxy = am.Galaxy(redshift=0.5)

            potential = galaxy.potential_from_grid(grid=sub_grid_7x7)

            assert (potential.in_1d == np.zeros(shape=sub_grid_7x7.shape[0])).all()

        def test__using_no_mass_profiles__check_reshaping_decorator_of_returned_potential(
            self, sub_grid_7x7
        ):
            galaxy = am.Galaxy(redshift=0.5)

            potential = galaxy.potential_from_grid(grid=sub_grid_7x7)

            assert (potential.in_2d_binned == np.zeros(shape=(7, 7))).all()

            potential = galaxy.potential_from_grid(grid=sub_grid_7x7)

            assert (potential.in_1d == np.zeros(shape=sub_grid_7x7.shape[0])).all()

            potential = galaxy.potential_from_grid(grid=sub_grid_7x7)

            assert (potential.in_1d_binned == np.zeros(shape=sub_grid_7x7.shape[0] // 4)).all()

        def test__galaxies_with_x1_and_x2_mass_profiles__potential_is_same_individual_profiles(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):
            mp_potential = mp_0.potential_from_grid(grid=aa.grid.manual_2d([[[1.05, -0.55]]]))

            gal_mp_potential = gal_x1_mp.potential_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )

            assert mp_potential == gal_mp_potential

            mp_potential = mp_0.potential_from_grid(grid=aa.grid.manual_2d([[[1.05, -0.55]]]))
            mp_potential += mp_1.potential_from_grid(grid=aa.grid.manual_2d([[[1.05, -0.55]]]))

            gal_potential = gal_x2_mp.potential_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )

            assert mp_potential == gal_potential

        def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper_by_binning_sum_of_mass_profile_values(
            self, sub_grid_7x7, gal_x2_mp
        ):
            mp_0_potential = gal_x2_mp.mass_profile_0.potential_from_grid(
                grid=sub_grid_7x7
            )

            mp_1_potential = gal_x2_mp.mass_profile_1.potential_from_grid(
                grid=sub_grid_7x7
            )

            mp_potential = mp_0_potential + mp_1_potential

            mp_potential_0 = (
                mp_potential[0] + mp_potential[1] + mp_potential[2] + mp_potential[3]
            ) / 4.0

            mp_potential_1 = (
                mp_potential[4] + mp_potential[5] + mp_potential[6] + mp_potential[7]
            ) / 4.0

            gal_potential = gal_x2_mp.potential_from_grid(grid=sub_grid_7x7)

            assert gal_potential.in_1d_binned[0] == mp_potential_0
            assert gal_potential.in_1d_binned[1] == mp_potential_1

    class TestDeflectionAngles:
        def test__no_mass_profiles__deflections_returned_as_0s_of_shape_grid(
            self, sub_grid_7x7
        ):
            galaxy = am.Galaxy(redshift=0.5)

            deflections = galaxy.deflections_from_grid(grid=sub_grid_7x7)

            assert (deflections.in_1d == np.zeros(shape=(sub_grid_7x7.shape[0], 2))).all()

        def test__using_no_mass_profiles__check_reshaping_decorator_of_returned_deflections(
            self, sub_grid_7x7
        ):
            galaxy = am.Galaxy(redshift=0.5)

            deflections = galaxy.deflections_from_grid(grid=sub_grid_7x7)

            assert (deflections.in_2d_binned == np.zeros(shape=(7, 7, 2))).all()

            deflections = galaxy.deflections_from_grid(grid=sub_grid_7x7)

            assert (deflections.in_1d == np.zeros(shape=(sub_grid_7x7.shape[0], 2))).all()

            deflections = galaxy.deflections_from_grid(grid=sub_grid_7x7)

            assert (
                deflections.in_1d_binned == np.zeros(shape=(sub_grid_7x7.shape[0] // 4, 2))
            ).all()

        def test__galaxies_with_x1_and_x2_mass_profiles__deflections_is_same_individual_profiles(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):
            mp_deflections = mp_0.deflections_from_grid(grid=aa.grid.manual_2d([[[1.05, -0.55]]]))

            gal_mp_deflections = gal_x1_mp.deflections_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )

            assert (mp_deflections == gal_mp_deflections).all()

            mp_deflections = mp_0.deflections_from_grid(grid=aa.grid.manual_2d([[[1.05, -0.55]]]))
            mp_deflections += mp_1.deflections_from_grid(grid=aa.grid.manual_2d([[[1.05, -0.55]]]))

            gal_deflections = gal_x2_mp.deflections_from_grid(
                grid=aa.grid.manual_2d([[[1.05, -0.55]]])
            )

            assert (mp_deflections == gal_deflections).all()

        def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper_by_binning_sum_of_mass_profile_values(
            self, sub_grid_7x7, gal_x2_mp
        ):
            mp_0_deflections = gal_x2_mp.mass_profile_0.deflections_from_grid(
                grid=sub_grid_7x7
            )

            mp_1_deflections = gal_x2_mp.mass_profile_1.deflections_from_grid(
                grid=sub_grid_7x7
            )

            mp_deflections = mp_0_deflections + mp_1_deflections

            mp_deflections_y_0 = (
                mp_deflections[0, 0]
                + mp_deflections[1, 0]
                + mp_deflections[2, 0]
                + mp_deflections[3, 0]
            ) / 4.0

            mp_deflections_y_1 = (
                mp_deflections[4, 0]
                + mp_deflections[5, 0]
                + mp_deflections[6, 0]
                + mp_deflections[7, 0]
            ) / 4.0

            gal_deflections = gal_x2_mp.deflections_from_grid(grid=sub_grid_7x7)

            assert gal_deflections.in_1d_binned[0, 0] == mp_deflections_y_0
            assert gal_deflections.in_1d_binned[1, 0] == mp_deflections_y_1

            mp_deflections_x_0 = (
                mp_deflections[0, 1]
                + mp_deflections[1, 1]
                + mp_deflections[2, 1]
                + mp_deflections[3, 1]
            ) / 4.0

            mp_deflections_x_1 = (
                mp_deflections[4, 1]
                + mp_deflections[5, 1]
                + mp_deflections[6, 1]
                + mp_deflections[7, 1]
            ) / 4.0

            gal_deflections = gal_x2_mp.deflections_from_grid(grid=sub_grid_7x7)

            assert gal_deflections.in_1d_binned[0, 1] == mp_deflections_x_0
            assert gal_deflections.in_1d_binned[1, 1] == mp_deflections_x_1

    class TestMassWithin:
        def test__two_profile_galaxy__is_sum_of_individual_profiles(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):
            radius = am.dim.Length(0.5, "arcsec")

            mp_mass = mp_0.mass_within_circle_in_units(
                radius=radius, unit_mass="angular"
            )

            gal_mass = gal_x1_mp.mass_within_circle_in_units(
                radius=radius, unit_mass="angular"
            )

            assert mp_mass == gal_mass

            mp_mass = mp_0.mass_within_ellipse_in_units(
                major_axis=radius, unit_mass="angular"
            )
            mp_mass += mp_1.mass_within_ellipse_in_units(
                major_axis=radius, unit_mass="angular"
            )

            gal_mass = gal_x2_mp.mass_within_ellipse_in_units(
                major_axis=radius, unit_mass="angular"
            )

            assert mp_mass == gal_mass

        def test__radius_unit_conversions__multiply_by_kpc_per_arcsec(
            self, mp_0, gal_x1_mp
        ):
            cosmology = mock_cosmology.MockCosmology(
                arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=1.0
            )

            radius = am.dim.Length(0.5, "arcsec")

            mp_mass_arcsec = mp_0.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_profile=0.5,
                redshift_source=1.0,
                cosmology=cosmology,
            )

            gal_mass_arcsec = gal_x1_mp.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_source=1.0,
                cosmology=cosmology,
            )
            assert mp_mass_arcsec == gal_mass_arcsec

            radius = am.dim.Length(0.5, "kpc")

            mp_mass_kpc = mp_0.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_profile=0.5,
                redshift_source=1.0,
                cosmology=cosmology,
            )

            gal_mass_kpc = gal_x1_mp.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_source=1.0,
                cosmology=cosmology,
            )
            assert mp_mass_kpc == gal_mass_kpc

        def test__mass_unit_conversions__same_as_individual_profile(
            self, mp_0, gal_x1_mp
        ):
            cosmology = mock_cosmology.MockCosmology(
                arcsec_per_kpc=1.0, kpc_per_arcsec=1.0, critical_surface_density=2.0
            )

            radius = am.dim.Length(0.5, "arcsec")

            mp_mass_angular = mp_0.mass_within_ellipse_in_units(
                major_axis=radius,
                unit_mass="angular",
                redshift_profile=0.5,
                redshift_source=1.0,
                cosmology=cosmology,
            )

            gal_mass_angular = gal_x1_mp.mass_within_ellipse_in_units(
                major_axis=radius,
                unit_mass="angular",
                redshift_source=1.0,
                cosmology=cosmology,
            )
            assert mp_mass_angular == gal_mass_angular

            mp_mass_sol = mp_0.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_profile=0.5,
                redshift_source=1.0,
                cosmology=cosmology,
            )

            gal_mass_sol = gal_x1_mp.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_source=1.0,
                cosmology=cosmology,
            )
            assert mp_mass_sol == gal_mass_sol

        def test__no_mass_profile__returns_none(self):
            gal_no_mp = am.Galaxy(
                redshift=0.5, light=am.lp.SphericalSersic()
            )

            assert (
                gal_no_mp.mass_within_circle_in_units(
                    radius=1.0, critical_surface_density=1.0
                )
                == None
            )
            assert (
                gal_no_mp.mass_within_ellipse_in_units(
                    major_axis=1.0, critical_surface_density=1.0
                )
                == None
            )

    class TestSymmetricProfiles:
        def test_1d_symmetry(self):
            mp_0 = am.mp.EllipticalIsothermal(
                axis_ratio=0.5, phi=45.0, einstein_radius=1.0
            )

            mp_1 = am.mp.EllipticalIsothermal(
                centre=(100, 0), axis_ratio=0.5, phi=45.0, einstein_radius=1.0
            )

            gal_x4_mp = am.Galaxy(
                redshift=0.5, mass_profile_0=mp_0, mass_profile_1=mp_1
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=aa.grid.manual_2d([[[1.0, 0.0]]])
            ) == gal_x4_mp.convergence_from_grid(grid=aa.grid.manual_2d([[[99.0, 0.0]]]))

            assert gal_x4_mp.convergence_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 0.0]]])
            ) == gal_x4_mp.convergence_from_grid(grid=aa.grid.manual_2d([[[51.0, 0.0]]]))

            assert gal_x4_mp.potential_from_grid(
                grid=aa.grid.manual_2d([[[1.0, 0.0]]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=aa.grid.manual_2d([[[99.0, 0.0]]])), 1e-6
            )

            assert gal_x4_mp.potential_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 0.0]]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=aa.grid.manual_2d([[[51.0, 0.0]]])), 1e-6
            )

            assert gal_x4_mp.deflections_from_grid(
                grid=aa.grid.manual_2d([[[1.0, 0.0]]])
            ) == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[99.0, 0.0]]])), 1e-6
            )

            assert gal_x4_mp.deflections_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 0.0]]])
            ) == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[51.0, 0.0]]])), 1e-6
            )

        def test_2d_symmetry(self):
            mp_0 = am.mp.SphericalIsothermal(einstein_radius=1.0)

            mp_1 = am.mp.SphericalIsothermal(
                centre=(100, 0), einstein_radius=1.0
            )

            mp_2 = am.mp.SphericalIsothermal(
                centre=(0, 100), einstein_radius=1.0
            )

            mp_3 = am.mp.SphericalIsothermal(
                centre=(100, 100), einstein_radius=1.0
            )

            gal_x4_mp = am.Galaxy(
                redshift=0.5,
                mass_profile_0=mp_0,
                mass_profile_1=mp_1,
                mass_profile_2=mp_2,
                mass_profile_3=mp_3,
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 0.0]]])
            ) == pytest.approx(
                gal_x4_mp.convergence_from_grid(grid=aa.grid.manual_2d([[[51.0, 0.0]]])), 1e-5
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=aa.grid.manual_2d([[[0.0, 49.0]]])
            ) == pytest.approx(
                gal_x4_mp.convergence_from_grid(grid=aa.grid.manual_2d([[[0.0, 51.0]]])), 1e-5
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=aa.grid.manual_2d([[[100.0, 49.0]]])
            ) == pytest.approx(
                gal_x4_mp.convergence_from_grid(grid=aa.grid.manual_2d([[[100.0, 51.0]]])), 1e-5
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 49.0]]])
            ) == pytest.approx(
                gal_x4_mp.convergence_from_grid(grid=aa.grid.manual_2d([[[51.0, 51.0]]])), 1e-5
            )

            assert gal_x4_mp.potential_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 0.0]]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=aa.grid.manual_2d([[[51.0, 0.0]]])), 1e-5
            )

            assert gal_x4_mp.potential_from_grid(
                grid=aa.grid.manual_2d([[[0.0, 49.0]]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=aa.grid.manual_2d([[[0.0, 51.0]]])), 1e-5
            )

            assert gal_x4_mp.potential_from_grid(
                grid=aa.grid.manual_2d([[[100.0, 49.0]]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=aa.grid.manual_2d([[[100.0, 51.0]]])), 1e-5
            )

            assert gal_x4_mp.potential_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 49.0]]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=aa.grid.manual_2d([[[51.0, 51.0]]])), 1e-5
            )

            assert -1.0 * gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[49.0, 0.0]]]))[
                0, 0
            ] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[51.0, 0.0]]]))[0, 0],
                1e-5,
            )

            assert 1.0 * gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[0.0, 49.0]]]))[
                0, 0
            ] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[0.0, 51.0]]]))[0, 0],
                1e-5,
            )

            assert 1.0 * gal_x4_mp.deflections_from_grid(
                grid=aa.grid.manual_2d([[[100.0, 49.0]]])
            )[0, 0] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[100.0, 51.0]]]))[0, 0],
                1e-5,
            )

            assert -1.0 * gal_x4_mp.deflections_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 49.0]]])
            )[0, 0] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[51.0, 51.0]]]))[0, 0],
                1e-5,
            )

            assert 1.0 * gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[49.0, 0.0]]]))[
                0, 1
            ] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[51.0, 0.0]]]))[0, 1],
                1e-5,
            )

            assert -1.0 * gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[0.0, 49.0]]]))[
                0, 1
            ] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[0.0, 51.0]]]))[0, 1],
                1e-5,
            )

            assert -1.0 * gal_x4_mp.deflections_from_grid(
                grid=aa.grid.manual_2d([[[100.0, 49.0]]])
            )[0, 1] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[100.0, 51.0]]]))[0, 1],
                1e-5,
            )

            assert -1.0 * gal_x4_mp.deflections_from_grid(
                grid=aa.grid.manual_2d([[[49.0, 49.0]]])
            )[0, 1] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=aa.grid.manual_2d([[[51.0, 51.0]]]))[0, 1],
                1e-5,
            )

    class TestEinsteinRadiiMass:
        def test__x2_sis_different_einstein_radii_and_mass__einstein_radii_and_mass_are_sum(
            self
        ):
            mp_0 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            mp_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=0.5
            )

            gal_x2_mp = am.Galaxy(redshift=0.5, mass_0=mp_0, mass_1=mp_1)

            assert gal_x2_mp.einstein_radius_in_units(unit_length="arcsec") == 1.5
            assert gal_x2_mp.einstein_mass_in_units(unit_mass="angular") == np.pi * (
                1.0 + 0.5 ** 2.0
            )

        def test__includes_shear__does_not_impact_values(self):
            mp_0 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            shear = am.mp.ExternalShear()

            gal_shear = am.Galaxy(redshift=0.5, mass_0=mp_0, shear=shear)

            assert gal_shear.einstein_radius_in_units(unit_length="arcsec") == 1.0
            assert gal_shear.einstein_mass_in_units(unit_mass="angular") == np.pi

    class TestDeflectionAnglesviaPotential(object):
        def test__compare_galaxy_deflections_via_potential_and_calculation(self):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )

            galaxy = am.Galaxy(mass_1=mass_profile_1, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(10, 10), pixel_scales=0.05, sub_size=1
            )

            deflections_via_calculation = galaxy.deflections_from_grid(grid=grid)

            deflections_via_potential = galaxy.deflections_via_potential_from_grid(
                grid=grid
            )

            mean_error = np.mean(
                deflections_via_potential - deflections_via_calculation
            )

            assert mean_error < 1e-4

        def test__compare_two_component_galaxy_deflections_via_potential_and_calculation(
            self
        ):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            mass_profile_2 = am.mp.SphericalIsothermal(
                centre=(1.0, 1.0), einstein_radius=1.0
            )

            galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(10, 10), pixel_scales=0.05, sub_size=1
            )

            deflections_via_calculation = galaxy.deflections_from_grid(grid=grid)

            deflections_via_potential = galaxy.deflections_via_potential_from_grid(
                grid=grid
            )

            mean_error = np.mean(
                deflections_via_potential - deflections_via_calculation
            )

            assert mean_error < 1e-4

        def test__galaxies_with_x1_and_x2_mass_profiles__deflections_via_potential_is_same_individual_profiles(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):
            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.05, sub_size=2
            )

            mp_deflections = mp_0.deflections_via_potential_from_grid(grid=grid)

            gal_mp_deflections = gal_x1_mp.deflections_via_potential_from_grid(
                grid=grid
            )

            mean_error = np.mean(mp_deflections - gal_mp_deflections)

            assert mean_error < 1e-4

            mp_deflections = mp_0.deflections_via_potential_from_grid(grid=grid)
            mp_deflections += mp_1.deflections_via_potential_from_grid(grid=grid)

            gal_deflections = gal_x2_mp.deflections_via_potential_from_grid(grid=grid)

            mean_error = np.mean(mp_deflections - gal_deflections)

            assert mean_error < 1e-4

    class TestJacobian(object):
        def test__jacobian_components(self):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )

            galaxy = am.Galaxy(mass_1=mass_profile_1, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(100, 100), pixel_scales=0.05, sub_size=1
            )

            jacobian = galaxy.jacobian_from_grid(grid=grid)

            A_12 = jacobian[0][1]
            A_21 = jacobian[1][0]

            mean_error = np.mean(A_12 - A_21)

            assert mean_error < 1e-4

            grid = aa.grid.uniform(
                shape_2d=(100, 100), pixel_scales=0.05, sub_size=2
            )

            jacobian = galaxy.jacobian_from_grid(grid=grid)

            A_12 = jacobian[0][1]
            A_21 = jacobian[1][0]

            mean_error = np.mean(A_12 - A_21)

            assert mean_error < 1e-4

        def test__jacobian_components__two_component_galaxy(self):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            mass_profile_2 = am.mp.SphericalIsothermal(
                centre=(1.0, 1.0), einstein_radius=1.0
            )

            galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(100, 100), pixel_scales=0.05, sub_size=1
            )

            jacobian = galaxy.jacobian_from_grid(grid=grid)

            A_12 = jacobian[0][1]
            A_21 = jacobian[1][0]

            mean_error = np.mean(A_12 - A_21)

            assert mean_error < 1e-4

            grid = aa.grid.uniform(
                shape_2d=(100, 100), pixel_scales=0.05, sub_size=2
            )

            jacobian = galaxy.jacobian_from_grid(grid=grid)

            A_12 = jacobian[0][1]
            A_21 = jacobian[1][0]

            mean_error = np.mean(A_12 - A_21)

            assert mean_error < 1e-4

    class TestConvergenceviaJacobian(object):
        def test__compare_galaxy_convergence_via_jacobian_and_calculation(self):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )

            galaxy = am.Galaxy(mass_1=mass_profile_1, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.05, sub_size=1
            )

            convergence_via_calculation = galaxy.convergence_from_grid(grid=grid)

            convergence_via_jacobian = galaxy.convergence_via_jacobian_from_grid(
                grid=grid
            )

            mean_error = np.mean(convergence_via_jacobian - convergence_via_calculation)

            assert mean_error < 1e-1

        def test__compare_two_component_galaxy_convergence_via_jacobian_and_calculation(
            self
        ):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            mass_profile_2 = am.mp.SphericalIsothermal(
                centre=(1.0, 1.0), einstein_radius=1.0
            )

            galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.05, sub_size=1
            )

            convergence_via_calculation = galaxy.convergence_from_grid(grid=grid)

            convergence_via_jacobian = galaxy.convergence_via_jacobian_from_grid(
                grid=grid
            )

            mean_error = np.mean(convergence_via_jacobian - convergence_via_calculation)

            assert mean_error < 1e-1

        def test__galaxies_with_x1_and_x2_mass_profiles__convergence_via_jacobian_is_same_individual_profiles(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):
            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.05, sub_size=2
            )

            mp_convergence = mp_0.convergence_via_jacobian_from_grid(grid=grid)

            gal_mp_convergence = gal_x1_mp.convergence_via_jacobian_from_grid(grid=grid)

            mean_error = np.mean(mp_convergence - gal_mp_convergence)

            assert mean_error < 1e-4

            mp_convergence = mp_0.convergence_via_jacobian_from_grid(grid=grid)
            mp_convergence += mp_1.convergence_via_jacobian_from_grid(grid=grid)

            gal_convergence = gal_x2_mp.convergence_via_jacobian_from_grid(grid=grid)

            mean_error = np.mean(mp_convergence - gal_convergence)

            assert mean_error < 1e-4

    class TestShearviaJacobian(object):
        def test__galaxies_with_x1_and_x2_mass_profiles__shear_via_jacobian_is_same_individual_profiles(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):
            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.05, sub_size=2
            )

            mp_shear = mp_0.shear_via_jacobian_from_grid(grid=grid)

            gal_mp_shear = gal_x1_mp.shear_via_jacobian_from_grid(grid=grid)

            mean_error = np.mean(mp_shear - gal_mp_shear)

            assert mean_error < 1e-4

            mp_shear = mp_0.shear_via_jacobian_from_grid(grid=grid)
            mp_shear += mp_1.shear_via_jacobian_from_grid(grid=grid)

            gal_shear = gal_x2_mp.shear_via_jacobian_from_grid(grid=grid)

            mean_error = np.mean(mp_shear - gal_shear)

            assert mean_error < 1e-4

    class TestMagnification(object):
        def test__compare_magnification_from_eigen_values_and_from_determinant__two_component_galaxy(
            self
        ):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            mass_profile_2 = am.mp.SphericalIsothermal(
                centre=(1.0, 1.0), einstein_radius=1.0
            )
            galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(100, 100), pixel_scales=0.05, sub_size=1
            )

            magnification_via_determinant = galaxy.magnification_from_grid(grid=grid)

            tangential_eigen_value = galaxy.tangential_eigen_value_from_grid(grid=grid)

            radal_eigen_value = galaxy.radial_eigen_value_from_grid(grid=grid)

            magnification_via_eigen_values = 1 / (
                tangential_eigen_value * radal_eigen_value
            )

            mean_error = np.mean(
                magnification_via_determinant - magnification_via_eigen_values
            )

            assert mean_error < 1e-4

            grid = aa.grid.uniform(
                shape_2d=(100, 100), pixel_scales=0.05, sub_size=2
            )

            magnification_via_determinant = galaxy.magnification_from_grid(grid=grid)

            tangential_eigen_value = galaxy.tangential_eigen_value_from_grid(grid=grid)

            radal_eigen_value = galaxy.radial_eigen_value_from_grid(grid=grid)

            magnification_via_eigen_values = 1 / (
                tangential_eigen_value * radal_eigen_value
            )

            mean_error = np.mean(
                magnification_via_determinant - magnification_via_eigen_values
            )

            assert mean_error < 1e-4

        def test__compare_magnification_from_determinant_and_from_convergence_and_shear__two_component_galaxy(
            self
        ):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            mass_profile_2 = am.mp.SphericalIsothermal(
                centre=(1.0, 1.0), einstein_radius=1.0
            )
            galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(100, 100), pixel_scales=0.05, sub_size=1
            )

            magnification_via_determinant = galaxy.magnification_from_grid(grid=grid)

            convergence = galaxy.convergence_via_jacobian_from_grid(grid=grid)

            shear = galaxy.shear_via_jacobian_from_grid(grid=grid)

            magnification_via_convergence_and_shear = 1 / (
                (1 - convergence) ** 2 - shear ** 2
            )

            mean_error = np.mean(
                magnification_via_determinant - magnification_via_convergence_and_shear
            )

            assert mean_error < 1e-4

            grid = aa.grid.uniform(
                shape_2d=(100, 100), pixel_scales=0.05, sub_size=2
            )

            magnification_via_determinant = galaxy.magnification_from_grid(grid=grid)

            convergence = galaxy.convergence_via_jacobian_from_grid(grid=grid)

            shear = galaxy.shear_via_jacobian_from_grid(grid=grid)

            magnification_via_convergence_and_shear = 1 / (
                (1 - convergence) ** 2 - shear ** 2
            )

            mean_error = np.mean(
                magnification_via_determinant - magnification_via_convergence_and_shear
            )

            assert mean_error < 1e-4

    class TestCriticalCurvesandCaustics(object):
        def test__compare_tangential_critical_curves_from_magnification_and_eigen_values__two_component_galaxy(
            self
        ):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            mass_profile_2 = am.mp.SphericalIsothermal(
                centre=(0.2, 0.2), einstein_radius=1.0
            )

            galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.25, sub_size=1
            )

            critical_curve_tangential_from_magnification = critical_curve_via_magnification_from_galaxy_and_grid(
                galaxy=galaxy, grid=grid
            )[
                0
            ]

            critical_curve_tangential_from_tangential_eigen_values = galaxy.critical_curves_from_grid(
                grid=grid
            )[
                0
            ]

            assert (
                critical_curve_tangential_from_tangential_eigen_values
                == pytest.approx(critical_curve_tangential_from_magnification, 5e-1)
            )

            grid = aa.grid.uniform(
                shape_2d=(50, 50), pixel_scales=0.5, sub_size=2
            )

            critical_curve_tangential_from_magnification = critical_curve_via_magnification_from_galaxy_and_grid(
                galaxy=galaxy, grid=grid
            )[
                0
            ]

            critical_curve_tangential_from_tangential_eigen_values = galaxy.critical_curves_from_grid(
                grid=grid
            )[
                0
            ]

            assert (
                critical_curve_tangential_from_tangential_eigen_values
                == pytest.approx(critical_curve_tangential_from_magnification, 5e-1)
            )

        def test__compare_radial_critical_curves_from_magnification_and_eigen_values__reg_grid_two_component_galaxy(
            self
        ):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            mass_profile_2 = am.mp.SphericalIsothermal(
                centre=(0.4, 0.4), einstein_radius=1.0
            )

            galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.5, sub_size=1
            )

            critical_curve_radial_from_magnification = critical_curve_via_magnification_from_galaxy_and_grid(
                galaxy=galaxy, grid=grid
            )[
                1
            ]

            critical_curve_radial_from_tangential_eigen_values = galaxy.critical_curves_from_grid(
                grid=grid
            )[
                1
            ]

            assert sum(
                critical_curve_radial_from_tangential_eigen_values
            ) == pytest.approx(sum(critical_curve_radial_from_magnification), 5e-1)

            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.5, sub_size=2
            )

            critical_curve_radial_from_magnification = critical_curve_via_magnification_from_galaxy_and_grid(
                galaxy=galaxy, grid=grid
            )[
                1
            ]

            critical_curve_radial_from_tangential_eigen_values = galaxy.critical_curves_from_grid(
                grid=grid
            )[
                1
            ]

            assert sum(
                critical_curve_radial_from_tangential_eigen_values
            ) == pytest.approx(sum(critical_curve_radial_from_magnification), 5e-1)

        def test__compare_tangential_caustic_from_magnification_and_tangential_eigen_values__two_component_galaxy(
            self
        ):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            mass_profile_2 = am.mp.SphericalIsothermal(
                centre=(0.2, 0.2), einstein_radius=1.0
            )

            galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.5, sub_size=1
            )

            caustic_tangential_from_magnification = caustics_via_magnification_from_galaxy_and_grid(
                galaxy=galaxy, grid=grid
            )[
                0
            ]

            caustic_tangential_from_tangential_eigen_values = galaxy.caustics_from_grid(
                grid=grid
            )[0]

            assert caustic_tangential_from_tangential_eigen_values == pytest.approx(
                caustic_tangential_from_magnification, 5e-1
            )

            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.5, sub_size=2
            )

            caustic_tangential_from_magnification = caustics_via_magnification_from_galaxy_and_grid(
                galaxy=galaxy, grid=grid
            )[
                0
            ]

            caustic_tangential_from_tangential_eigen_values = galaxy.caustics_from_grid(
                grid=grid
            )[0]

            assert caustic_tangential_from_tangential_eigen_values == pytest.approx(
                caustic_tangential_from_magnification, 5e-1
            )

        def test__compare_radial_caustic_from_magnification_and_tangential_eigen_values__two_component_galaxy(
            self
        ):
            mass_profile_1 = am.mp.SphericalIsothermal(
                centre=(0.0, 0.0), einstein_radius=1.0
            )
            mass_profile_2 = am.mp.SphericalIsothermal(
                centre=(0.2, 0.2), einstein_radius=1.0
            )

            galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

            grid = aa.grid.uniform(
                shape_2d=(20, 20), pixel_scales=0.25, sub_size=2
            )

            caustic_radial_from_magnification = caustics_via_magnification_from_galaxy_and_grid(
                galaxy=galaxy, grid=grid
            )[
                1
            ]
            caustic_radial_from_tangential_eigen_values = galaxy.caustics_from_grid(
                grid=grid
            )[1]

            assert sum(caustic_radial_from_tangential_eigen_values) == pytest.approx(
                sum(caustic_radial_from_magnification), 5e-1
            )


class TestMassAndLightProfiles(object):
    def test_single_profile(self, lmp_0):
        gal_x1_lmp = am.Galaxy(redshift=0.5, profile=lmp_0)

        assert 1 == len(gal_x1_lmp.light_profiles)
        assert 1 == len(gal_x1_lmp.mass_profiles)

        assert gal_x1_lmp.mass_profiles[0] == lmp_0
        assert gal_x1_lmp.light_profiles[0] == lmp_0

    def test_multiple_profile(self, lmp_0, lp_0, mp_0):
        gal_multi_profiles = am.Galaxy(
            redshift=0.5, profile=lmp_0, light=lp_0, sie=mp_0
        )

        assert 2 == len(gal_multi_profiles.light_profiles)
        assert 2 == len(gal_multi_profiles.mass_profiles)


class TestSummarizeInUnits(object):
    def test__galaxy_with_two_light_and_mass_profiles(self, lp_0, lp_1, mp_0, mp_1):
        gal_summarize = am.Galaxy(
            redshift=0.5,
            light_profile_0=lp_0,
            light_profile_1=lp_1,
            mass_profile_0=mp_0,
            mass_profile_1=mp_1,
        )

        summary_text = gal_summarize.summarize_in_units(
            radii=[am.dim.Length(10.0), am.dim.Length(500.0)],
            whitespace=50,
            unit_length="arcsec",
            unit_luminosity="eps",
            unit_mass="angular",
        )

        i = 0

        assert summary_text[i] == "Galaxy\n"
        i += 1
        assert (
            summary_text[i] == "redshift                                          0.50"
        )
        i += 1
        assert summary_text[i] == "\nGALAXY LIGHT\n\n"
        i += 1
        assert (
            summary_text[i]
            == "luminosity_within_10.00_arcsec                    1.8854e+02 eps"
        )
        i += 1
        assert (
            summary_text[i]
            == "luminosity_within_500.00_arcsec                   1.9573e+02 eps"
        )
        i += 1
        assert summary_text[i] == "\nLIGHT PROFILES:\n\n"
        i += 1
        assert summary_text[i] == "Light Profile = SphericalSersic\n"
        i += 1
        assert (
            summary_text[i]
            == "luminosity_within_10.00_arcsec                    6.2848e+01 eps"
        )
        i += 1
        assert (
            summary_text[i]
            == "luminosity_within_500.00_arcsec                   6.5243e+01 eps"
        )
        i += 1
        assert summary_text[i] == "\n"
        i += 1
        assert summary_text[i] == "Light Profile = SphericalSersic\n"
        i += 1
        assert (
            summary_text[i]
            == "luminosity_within_10.00_arcsec                    1.2570e+02 eps"
        )
        i += 1
        assert (
            summary_text[i]
            == "luminosity_within_500.00_arcsec                   1.3049e+02 eps"
        )
        i += 1
        assert summary_text[i] == "\n"
        i += 1
        assert summary_text[i] == "\nGALAXY MASS\n\n"
        i += 1
        assert (
            summary_text[i]
            == "einstein_radius                                   3.00 arcsec"
        )
        i += 1
        assert (
            summary_text[i]
            == "einstein_mass                                     1.5708e+01 angular"
        )
        i += 1
        assert (
            summary_text[i]
            == "mass_within_10.00_arcsec                          9.4248e+01 angular"
        )
        i += 1
        assert (
            summary_text[i]
            == "mass_within_500.00_arcsec                         4.7124e+03 angular"
        )
        i += 1
        assert summary_text[i] == "\nMASS PROFILES:\n\n"
        i += 1
        assert summary_text[i] == "Mass Profile = SphericalIsothermal\n"
        i += 1
        assert (
            summary_text[i]
            == "einstein_radius                                   1.00 arcsec"
        )
        i += 1
        assert (
            summary_text[i]
            == "einstein_mass                                     3.1416e+00 angular"
        )
        i += 1
        assert (
            summary_text[i]
            == "mass_within_10.00_arcsec                          3.1416e+01 angular"
        )
        i += 1
        assert (
            summary_text[i]
            == "mass_within_500.00_arcsec                         1.5708e+03 angular"
        )
        i += 1
        assert summary_text[i] == "\n"
        i += 1
        assert summary_text[i] == "Mass Profile = SphericalIsothermal\n"
        i += 1
        assert (
            summary_text[i]
            == "einstein_radius                                   2.00 arcsec"
        )
        i += 1
        assert (
            summary_text[i]
            == "einstein_mass                                     1.2566e+01 angular"
        )
        i += 1
        assert (
            summary_text[i]
            == "mass_within_10.00_arcsec                          6.2832e+01 angular"
        )
        i += 1
        assert (
            summary_text[i]
            == "mass_within_500.00_arcsec                         3.1416e+03 angular"
        )
        i += 1
        assert summary_text[i] == "\n"
        i += 1


class TestHyperGalaxy(object):
    class TestContributionMaps(object):
        def test__model_image_all_1s__factor_is_0__contributions_all_1s(self):
            hyper_image = np.ones((3,))

            hyp = am.HyperGalaxy(contribution_factor=0.0)
            contribution_map = hyp.contribution_map_from_hyper_images(
                hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image
            )

            assert (contribution_map == np.ones((3,))).all()

        def test__different_values__factor_is_1__contributions_are_value_divided_by_factor_and_max(
            self
        ):
            hyper_image = np.array([0.5, 1.0, 1.5])

            hyp = am.HyperGalaxy(contribution_factor=1.0)
            contribution_map = hyp.contribution_map_from_hyper_images(
                hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image
            )

            assert (
                contribution_map
                == np.array([(0.5 / 1.5) / (1.5 / 2.5), (1.0 / 2.0) / (1.5 / 2.5), 1.0])
            ).all()

    class TestHyperNoiseMap(object):
        def test__contribution_all_1s__noise_factor_2__noise_adds_double(self):
            noise_map = np.array([1.0, 2.0, 3.0])
            contribution_map = np.ones((3, 1))

            hyper_galaxy = am.HyperGalaxy(
                contribution_factor=0.0, noise_factor=2.0, noise_power=1.0
            )

            hyper_noise_map = hyper_galaxy.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map
            )

            assert (hyper_noise_map == np.array([2.0, 4.0, 6.0])).all()

        def test__same_as_above_but_contributions_vary(self):
            noise_map = np.array([1.0, 2.0, 3.0])
            contribution_map = np.array([[0.0, 0.5, 1.0]])

            hyper_galaxy = am.HyperGalaxy(
                contribution_factor=0.0, noise_factor=2.0, noise_power=1.0
            )

            hyper_noise_map = hyper_galaxy.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map
            )

            assert (hyper_noise_map == np.array([0.0, 2.0, 6.0])).all()

        def test__same_as_above_but_change_noise_scale_terms(self):
            noise_map = np.array([1.0, 2.0, 3.0])
            contribution_map = np.array([[0.0, 0.5, 1.0]])

            hyper_galaxy = am.HyperGalaxy(
                contribution_factor=0.0, noise_factor=2.0, noise_power=2.0
            )

            hyper_noise_map = hyper_galaxy.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map
            )

            assert (hyper_noise_map == np.array([0.0, 2.0, 18.0])).all()


class TestBooleanProperties(object):
    def test_has_profile(self):
        assert am.Galaxy(redshift=0.5).has_profile is False
        assert (
            am.Galaxy(
                redshift=0.5, light_profile=am.lp.LightProfile()
            ).has_profile
            is True
        )
        assert (
            am.Galaxy(
                redshift=0.5, mass_profile=am.mp.MassProfile()
            ).has_profile
            is True
        )

    def test_has_light_profile(self):
        assert am.Galaxy(redshift=0.5).has_light_profile is False
        assert (
            am.Galaxy(
                redshift=0.5, light_profile=am.lp.LightProfile()
            ).has_light_profile
            is True
        )
        assert (
            am.Galaxy(
                redshift=0.5, mass_profile=am.mp.MassProfile()
            ).has_light_profile
            is False
        )

    def test_has_mass_profile(self):
        assert am.Galaxy(redshift=0.5).has_mass_profile is False
        assert (
            am.Galaxy(
                redshift=0.5, light_profile=am.lp.LightProfile()
            ).has_mass_profile
            is False
        )
        assert (
            am.Galaxy(
                redshift=0.5, mass_profile=am.mp.MassProfile()
            ).has_mass_profile
            is True
        )

    def test_has_redshift(self):
        assert am.Galaxy(redshift=0.1).has_redshift is True

    def test_has_pixelization(self):
        assert am.Galaxy(redshift=0.5).has_pixelization is False
        assert (
            am.Galaxy(
                redshift=0.5, pixelization=object(), regularization=object()
            ).has_pixelization
            is True
        )

    def test_has_regularization(self):
        assert am.Galaxy(redshift=0.5).has_regularization is False
        assert (
            am.Galaxy(
                redshift=0.5, pixelization=object(), regularization=object()
            ).has_regularization
            is True
        )

    def test_has_hyper_galaxy(self):
        assert am.Galaxy(redshift=0.5, hyper_galaxy=object()).has_hyper_galaxy is True

    def test__only_pixelization_raises_error(self):
        with pytest.raises(exc.GalaxyException):
            am.Galaxy(redshift=0.5, pixelization=object())

    def test__only_regularization_raises_error(self):
        with pytest.raises(exc.GalaxyException):
            am.Galaxy(redshift=0.5, regularization=object())


class TestGridBinning(object):
    def test__jacobian_sub_grid_binning_two_component_galaxy(self):
        mass_profile_1 = am.mp.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )
        mass_profile_2 = am.mp.SphericalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

        grid = aa.grid.uniform(
            shape_2d=(10, 10), pixel_scales=0.05, sub_size=2
        )

        jacobian = galaxy.jacobian_from_grid(grid=grid)

        jacobian_1st_pixel_binned_up = (
            jacobian[0][0][0]
            + jacobian[0][0][1]
            + jacobian[0][0][2]
            + jacobian[0][0][3]
        ) / 4

        assert jacobian[0][0].in_2d_binned.shape == (10, 10)
        assert jacobian[0][0].sub_shape_2d == (20, 20)
        assert jacobian[0][0].in_1d_binned[0] == pytest.approx(
            jacobian_1st_pixel_binned_up, 1e-4
        )

        jacobian_last_pixel_binned_up = (
            jacobian[0][0][399]
            + jacobian[0][0][398]
            + jacobian[0][0][397]
            + jacobian[0][0][396]
        ) / 4

        assert jacobian[0][0].in_1d_binned[99] == pytest.approx(
            jacobian_last_pixel_binned_up, 1e-4
        )

    def test_shear_via_jacobian(self):
        mass_profile_1 = am.mp.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )
        mass_profile_2 = am.mp.SphericalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

        grid = aa.grid.uniform(
            shape_2d=(10, 10), pixel_scales=0.05, sub_size=2
        )

        shear_via_jacobian = galaxy.shear_via_jacobian_from_grid(grid=grid)

        shear_1st_pixel_binned_up = (
            shear_via_jacobian[0]
            + shear_via_jacobian[1]
            + shear_via_jacobian[2]
            + shear_via_jacobian[3]
        ) / 4

        assert shear_via_jacobian.in_1d_binned[0] == pytest.approx(
            shear_1st_pixel_binned_up, 1e-4
        )

        shear_last_pixel_binned_up = (
            shear_via_jacobian[399]
            + shear_via_jacobian[398]
            + shear_via_jacobian[397]
            + shear_via_jacobian[396]
        ) / 4

        assert shear_via_jacobian.in_1d_binned[99] == pytest.approx(
            shear_last_pixel_binned_up, 1e-4
        )

    def test_tangential_eigen_values_sub_grid_binning_two_component_galaxy(self):
        mass_profile_1 = am.mp.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )
        mass_profile_2 = am.mp.SphericalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

        grid = aa.grid.uniform(
            shape_2d=(10, 10), pixel_scales=0.05, sub_size=2
        )

        tangential_eigen_values = galaxy.tangential_eigen_value_from_grid(grid=grid)

        first_pixel_binned_up = (
            tangential_eigen_values[0]
            + tangential_eigen_values[1]
            + tangential_eigen_values[2]
            + tangential_eigen_values[3]
        ) / 4

        assert tangential_eigen_values.in_1d_binned[0] == pytest.approx(
            first_pixel_binned_up, 1e-4
        )

        pixel_10000_from_av_sub_grid = (
            tangential_eigen_values[399]
            + tangential_eigen_values[398]
            + tangential_eigen_values[397]
            + tangential_eigen_values[396]
        ) / 4

        assert tangential_eigen_values.in_1d_binned[99] == pytest.approx(
            pixel_10000_from_av_sub_grid, 1e-4
        )

    def test_radial_eigen_values_sub_grid_binning_two_component_galaxy(self):
        mass_profile_1 = am.mp.SphericalIsothermal(
            centre=(0.0, 0.0), einstein_radius=1.0
        )
        mass_profile_2 = am.mp.SphericalIsothermal(
            centre=(1.0, 1.0), einstein_radius=1.0
        )

        galaxy = am.Galaxy(mass_1=mass_profile_1, mass_2=mass_profile_2, redshift=1)

        grid = aa.grid.uniform(
            shape_2d=(100, 100), pixel_scales=0.05, sub_size=2
        )

        radial_eigen_values = galaxy.radial_eigen_value_from_grid(grid=grid)

        first_pixel_binned_up = (
            radial_eigen_values[0]
            + radial_eigen_values[1]
            + radial_eigen_values[2]
            + radial_eigen_values[3]
        ) / 4

        assert radial_eigen_values.in_1d_binned[0] == pytest.approx(
            first_pixel_binned_up, 1e-4
        )

        pixel_10000_from_av_sub_grid = (
            radial_eigen_values[399]
            + radial_eigen_values[398]
            + radial_eigen_values[397]
            + radial_eigen_values[396]
        ) / 4

        assert radial_eigen_values.in_1d_binned[99] == pytest.approx(
            pixel_10000_from_av_sub_grid, 1e-4
        )