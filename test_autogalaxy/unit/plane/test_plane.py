import autogalaxy as ag
import numpy as np
import pytest
from autogalaxy import exc
from autogalaxy.plane import plane
from skimage import measure
from autoarray.mock import mock


def critical_curve_via_magnification_from_plane_and_grid(plane, grid):
    magnification = plane.magnification_from_grid(grid=grid)

    inverse_magnification = 1 / magnification

    critical_curves_indices = measure.find_contours(inverse_magnification.native, 0)

    no_critical_curves = len(critical_curves_indices)
    contours = []
    critical_curves = []

    for jj in np.arange(no_critical_curves):
        contours.append(critical_curves_indices[jj])
        contour_x, contour_y = contours[jj].T
        pixel_coord = np.stack((contour_x, contour_y), axis=-1)

        critical_curve = grid.mask.grid_scaled_from_grid_pixels_1d_for_marching_squares(
            grid_pixels_1d=pixel_coord, shape_native=magnification.sub_shape_native
        )

        critical_curve = np.array(grid=critical_curve)

        critical_curves.append(critical_curve)

    return critical_curves


def caustics_via_magnification_from_plane_and_grid(plane, grid):
    caustics = []

    critical_curves = critical_curve_via_magnification_from_plane_and_grid(
        plane=plane, grid=grid
    )

    for i in range(len(critical_curves)):
        critical_curve = critical_curves[i]

        deflections_1d = plane.deflections_from_grid(grid=critical_curve)

        caustic = critical_curve - deflections_1d

        caustics.append(caustic)

    return caustics


class TestAbstractPlane:
    class TestProperties:
        def test__has_light_profile(self):
            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_light_profile is False

            plane = ag.Plane(
                galaxies=[ag.Galaxy(redshift=0.5, light_profile=ag.lp.LightProfile())],
                redshift=None,
            )
            assert plane.has_light_profile is True

            plane = ag.Plane(
                galaxies=[
                    ag.Galaxy(redshift=0.5, light_profile=ag.lp.LightProfile()),
                    ag.Galaxy(redshift=0.5),
                ],
                redshift=None,
            )
            assert plane.has_light_profile is True

        def test__has_mass_profile(self):
            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_mass_profile is False

            plane = ag.Plane(
                galaxies=[ag.Galaxy(redshift=0.5, mass_profile=ag.mp.MassProfile())],
                redshift=None,
            )
            assert plane.has_mass_profile is True

            plane = ag.Plane(
                galaxies=[
                    ag.Galaxy(redshift=0.5, mass_profile=ag.mp.MassProfile()),
                    ag.Galaxy(redshift=0.5),
                ],
                redshift=None,
            )
            assert plane.has_mass_profile is True

        def test__has_pixelization(self):
            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_pixelization is False

            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=ag.pix.Pixelization(),
                regularization=ag.reg.Regularization(),
            )

            plane = ag.Plane(galaxies=[galaxy_pix], redshift=None)
            assert plane.has_pixelization is True

            plane = ag.Plane(
                galaxies=[galaxy_pix, ag.Galaxy(redshift=0.5)], redshift=None
            )
            assert plane.has_pixelization is True

        def test__has_regularization(self):
            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_regularization is False

            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=ag.pix.Pixelization(),
                regularization=ag.reg.Regularization(),
            )

            plane = ag.Plane(galaxies=[galaxy_pix], redshift=None)
            assert plane.has_regularization is True

            plane = ag.Plane(
                galaxies=[galaxy_pix, ag.Galaxy(redshift=0.5)], redshift=None
            )
            assert plane.has_regularization is True

        def test__has_hyper_galaxy(self):
            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_hyper_galaxy is False

            galaxy = ag.Galaxy(redshift=0.5, hyper_galaxy=ag.HyperGalaxy())

            plane = ag.Plane(galaxies=[galaxy], redshift=None)
            assert plane.has_hyper_galaxy is True

            plane = ag.Plane(galaxies=[galaxy, ag.Galaxy(redshift=0.5)], redshift=None)
            assert plane.has_hyper_galaxy is True

        def test__mass_profiles(self):

            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

            assert plane.mass_profiles == []

            sis_0 = ag.mp.SphericalIsothermal(einstein_radius=1.0)
            sis_1 = ag.mp.SphericalIsothermal(einstein_radius=2.0)
            sis_2 = ag.mp.SphericalIsothermal(einstein_radius=3.0)

            plane = ag.Plane(
                galaxies=[ag.Galaxy(redshift=0.5, mass_profile=sis_0)], redshift=None
            )
            assert plane.mass_profiles == [sis_0]

            plane = ag.Plane(
                galaxies=[
                    ag.Galaxy(redshift=0.5, mass_profile_0=sis_0, mass_profile_1=sis_1),
                    ag.Galaxy(redshift=0.5, mass_profile_0=sis_2, mass_profile_1=sis_1),
                ],
                redshift=None,
            )
            assert plane.mass_profiles == [sis_0, sis_1, sis_2, sis_1]

        def test__hyper_image_of_galaxy_with_pixelization(self):
            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=ag.pix.Pixelization(),
                regularization=ag.reg.Regularization(),
            )

            plane = ag.Plane(galaxies=[galaxy_pix], redshift=None)
            assert plane.hyper_galaxy_image_of_galaxy_with_pixelization is None

            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=ag.pix.Pixelization(),
                regularization=ag.reg.Regularization(),
                hyper_galaxy_image=1,
            )

            plane = ag.Plane(
                galaxies=[galaxy_pix, ag.Galaxy(redshift=0.5)], redshift=None
            )
            assert plane.hyper_galaxy_image_of_galaxy_with_pixelization == 1

            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

            assert plane.hyper_galaxy_image_of_galaxy_with_pixelization is None

    class TestPixelization:
        def test__no_galaxies_with_pixelizations_in_plane__returns_none(self):
            galaxy_no_pix = ag.Galaxy(redshift=0.5)

            plane = ag.Plane(galaxies=[galaxy_no_pix], redshift=None)

            assert plane.pixelization is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_mapper(self):
            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=1),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )

            plane = ag.Plane(galaxies=[galaxy_pix], redshift=None)

            assert plane.pixelization.value == 1

            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=2),
                regularization=mock.MockRegularization(matrix_shape=(2, 2)),
            )
            galaxy_no_pix = ag.Galaxy(redshift=0.5)

            plane = ag.Plane(galaxies=[galaxy_no_pix, galaxy_pix], redshift=None)

            assert plane.pixelization.value == 2

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(self):
            galaxy_pix_0 = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=1),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_pix_1 = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=2),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )

            plane = ag.Plane(galaxies=[galaxy_pix_0, galaxy_pix_1], redshift=None)

            with pytest.raises(exc.PixelizationException):
                print(plane.pixelization)

    class TestRegularization:
        def test__no_galaxies_with_regularizations_in_plane__returns_none(self):
            galaxy_no_pix = ag.Galaxy(redshift=0.5)

            plane = ag.Plane(galaxies=[galaxy_no_pix], redshift=None)

            assert plane.regularization is None

        def test__1_galaxy_in_plane__it_has_regularization__returns_regularization(
            self,
        ):
            galaxy_reg = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=1),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )

            plane = ag.Plane(galaxies=[galaxy_reg], redshift=None)

            assert plane.regularization.shape == (1, 1)

            galaxy_reg = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=1),
                regularization=mock.MockRegularization(matrix_shape=(2, 2)),
            )
            galaxy_no_reg = ag.Galaxy(redshift=0.5)

            plane = ag.Plane(galaxies=[galaxy_no_reg, galaxy_reg], redshift=None)

            assert plane.regularization.shape == (2, 2)

        def test__2_galaxies_in_plane__both_have_regularization__raises_error(self):
            galaxy_reg_0 = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=1),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_reg_1 = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=2),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )

            plane = ag.Plane(galaxies=[galaxy_reg_0, galaxy_reg_1], redshift=None)

            with pytest.raises(exc.PixelizationException):
                print(plane.regularization)

    class TestProfileGeometry:
        def test__extract_centres_of_all_light_profiles_of_all_galaxies(self):

            g0 = ag.Galaxy(
                redshift=0.5, light=ag.lp.SphericalGaussian(centre=(1.0, 1.0))
            )
            g1 = ag.Galaxy(
                redshift=0.5, light=ag.lp.SphericalGaussian(centre=(2.0, 2.0))
            )
            g2 = ag.Galaxy(
                redshift=0.5,
                light0=ag.lp.SphericalGaussian(centre=(3.0, 3.0)),
                light1=ag.lp.SphericalGaussian(centre=(4.0, 4.0)),
            )

            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

            assert plane.light_profile_centres == []

            plane = ag.Plane(galaxies=[g0], redshift=None)

            print(plane.light_profile_centres.in_grouped_list)

            assert plane.light_profile_centres.in_grouped_list == [[(1.0, 1.0)]]

            plane = ag.Plane(galaxies=[g1], redshift=None)

            assert plane.light_profile_centres.in_grouped_list == [[(2.0, 2.0)]]

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            assert plane.light_profile_centres.in_grouped_list == [
                [(1.0, 1.0)],
                [(2.0, 2.0)],
            ]

            plane = ag.Plane(galaxies=[g1, g0], redshift=None)

            assert plane.light_profile_centres.in_grouped_list == [
                [(2.0, 2.0)],
                [(1.0, 1.0)],
            ]

            plane = ag.Plane(
                galaxies=[g0, ag.Galaxy(redshift=0.5), g1, ag.Galaxy(redshift=0.5)],
                redshift=None,
            )

            assert plane.light_profile_centres.in_grouped_list == [
                [(1.0, 1.0)],
                [(2.0, 2.0)],
            ]

            plane = ag.Plane(
                galaxies=[g0, ag.Galaxy(redshift=0.5), g1, ag.Galaxy(redshift=0.5), g2],
                redshift=None,
            )

            assert plane.light_profile_centres.in_grouped_list == [
                [(1.0, 1.0)],
                [(2.0, 2.0)],
                [(3.0, 3.0), (4.0, 4.0)],
            ]

        def test__extract_centres_of_all_mass_profiles_of_all_galaxies__ignores_mass_sheets(
            self,
        ):

            g0 = ag.Galaxy(
                redshift=0.5, mass=ag.mp.SphericalIsothermal(centre=(1.0, 1.0))
            )
            g1 = ag.Galaxy(
                redshift=0.5, mass=ag.mp.SphericalIsothermal(centre=(2.0, 2.0))
            )
            g2 = ag.Galaxy(
                redshift=0.5,
                mass0=ag.mp.SphericalIsothermal(centre=(3.0, 3.0)),
                mass1=ag.mp.SphericalIsothermal(centre=(4.0, 4.0)),
            )

            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

            assert plane.mass_profile_centres == []

            plane = ag.Plane(galaxies=[g0], redshift=None)

            assert plane.mass_profile_centres.in_grouped_list == [[(1.0, 1.0)]]

            plane = ag.Plane(galaxies=[g1], redshift=None)

            assert plane.mass_profile_centres.in_grouped_list == [[(2.0, 2.0)]]

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            assert plane.mass_profile_centres.in_grouped_list == [
                [(1.0, 1.0)],
                [(2.0, 2.0)],
            ]

            plane = ag.Plane(galaxies=[g1, g0], redshift=None)

            assert plane.mass_profile_centres.in_grouped_list == [
                [(2.0, 2.0)],
                [(1.0, 1.0)],
            ]

            plane = ag.Plane(
                galaxies=[g0, ag.Galaxy(redshift=0.5), g1, ag.Galaxy(redshift=0.5)],
                redshift=None,
            )

            assert plane.mass_profile_centres.in_grouped_list == [
                [(1.0, 1.0)],
                [(2.0, 2.0)],
            ]

            plane = ag.Plane(
                galaxies=[g0, ag.Galaxy(redshift=0.5), g1, ag.Galaxy(redshift=0.5), g2],
                redshift=None,
            )

            assert plane.mass_profile_centres.in_grouped_list == [
                [(1.0, 1.0)],
                [(2.0, 2.0)],
                [(3.0, 3.0), (4.0, 4.0)],
            ]

            g0 = ag.Galaxy(
                redshift=0.5,
                mass=ag.mp.SphericalIsothermal(centre=(1.0, 1.0)),
                sheet=ag.mp.MassSheet(centre=(10.0, 10.0)),
            )

            plane = ag.Plane(
                galaxies=[
                    g0,
                    ag.Galaxy(redshift=0.5, sheet=ag.mp.MassSheet(centre=(10.0, 10.0))),
                    g1,
                    ag.Galaxy(redshift=0.5, sheet=ag.mp.MassSheet(centre=(10.0, 10.0))),
                    g2,
                ],
                redshift=None,
            )

            assert plane.mass_profile_centres.in_grouped_list == [
                [(1.0, 1.0)],
                [(2.0, 2.0)],
                [(3.0, 3.0), (4.0, 4.0)],
            ]

        def test__extracts_axis_ratio_of_all_mass_profiles_of_all_galaxies(self):

            g0 = ag.Galaxy(
                redshift=0.5,
                mass=ag.mp.EllipticalIsothermal(elliptical_comps=(0.0, 0.05263)),
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass=ag.mp.EllipticalIsothermal.from_axis_ratio_and_phi(axis_ratio=0.8),
            )
            g2 = ag.Galaxy(
                redshift=0.5,
                mass0=ag.mp.EllipticalIsothermal.from_axis_ratio_and_phi(
                    axis_ratio=0.7
                ),
                mass1=ag.mp.EllipticalIsothermal.from_axis_ratio_and_phi(
                    axis_ratio=0.6
                ),
            )

            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

            assert plane.mass_profile_axis_ratios == []

            plane = ag.Plane(galaxies=[g0], redshift=None)

            assert plane.mass_profile_axis_ratios.in_grouped_list[0][
                0
            ] == pytest.approx(0.9, 1.0e-4)

            plane = ag.Plane(galaxies=[g1], redshift=None)

            assert plane.mass_profile_axis_ratios.in_grouped_list[0][
                0
            ] == pytest.approx(0.8, 1.0e-4)

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            assert plane.mass_profile_axis_ratios.in_grouped_list[0][
                0
            ] == pytest.approx(0.9, 1.0e-4)
            assert plane.mass_profile_axis_ratios.in_grouped_list[1][
                0
            ] == pytest.approx(0.8, 1.0e-4)

            plane = ag.Plane(galaxies=[g1, g0], redshift=None)

            assert plane.mass_profile_axis_ratios.in_grouped_list[0][
                0
            ] == pytest.approx(0.8, 1.0e-4)
            assert plane.mass_profile_axis_ratios.in_grouped_list[1][
                0
            ] == pytest.approx(0.9, 1.0e-4)

            plane = ag.Plane(
                galaxies=[g0, ag.Galaxy(redshift=0.5), g1, ag.Galaxy(redshift=0.5)],
                redshift=None,
            )

            assert plane.mass_profile_axis_ratios.in_grouped_list[0][
                0
            ] == pytest.approx(0.9, 1.0e-4)
            assert plane.mass_profile_axis_ratios.in_grouped_list[1][
                0
            ] == pytest.approx(0.8, 1.0e-4)

            plane = ag.Plane(
                galaxies=[g0, ag.Galaxy(redshift=0.5), g1, ag.Galaxy(redshift=0.5), g2],
                redshift=None,
            )

            assert plane.mass_profile_axis_ratios.in_grouped_list[0][
                0
            ] == pytest.approx(0.9, 1.0e-4)
            assert plane.mass_profile_axis_ratios.in_grouped_list[1][
                0
            ] == pytest.approx(0.8, 1.0e-4)
            assert plane.mass_profile_axis_ratios.in_grouped_list[2][
                0
            ] == pytest.approx(0.7, 1.0e-4)
            assert plane.mass_profile_axis_ratios.in_grouped_list[2][
                1
            ] == pytest.approx(0.6, 1.0e-4)

        def test__extracts_phi_of_all_mass_profiles_of_all_galaxies(self):

            g0 = ag.Galaxy(
                redshift=0.5,
                mass=ag.mp.EllipticalIsothermal.from_axis_ratio_and_phi(
                    axis_ratio=0.1, phi=0.9
                ),
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass=ag.mp.EllipticalIsothermal.from_axis_ratio_and_phi(
                    axis_ratio=0.1, phi=0.8
                ),
            )
            g2 = ag.Galaxy(
                redshift=0.5,
                mass0=ag.mp.EllipticalIsothermal.from_axis_ratio_and_phi(
                    axis_ratio=0.1, phi=0.7
                ),
                mass1=ag.mp.EllipticalIsothermal.from_axis_ratio_and_phi(
                    axis_ratio=0.1, phi=0.6
                ),
            )

            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

            assert plane.mass_profile_phis == []

            plane = ag.Plane(galaxies=[g0], redshift=None)

            assert plane.mass_profile_phis.in_grouped_list[0][0] == pytest.approx(
                0.9, 1.0e-4
            )

            plane = ag.Plane(galaxies=[g1], redshift=None)

            assert plane.mass_profile_phis.in_grouped_list[0][0] == pytest.approx(
                0.8, 1.0e-4
            )

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            assert plane.mass_profile_phis.in_grouped_list[0][0] == pytest.approx(
                0.9, 1.0e-4
            )
            assert plane.mass_profile_phis.in_grouped_list[1][0] == pytest.approx(
                0.8, 1.0e-4
            )

            plane = ag.Plane(galaxies=[g1, g0], redshift=None)

            assert plane.mass_profile_phis.in_grouped_list[0][0] == pytest.approx(
                0.8, 1.0e-4
            )
            assert plane.mass_profile_phis.in_grouped_list[1][0] == pytest.approx(
                0.9, 1.0e-4
            )

            plane = ag.Plane(
                galaxies=[g0, ag.Galaxy(redshift=0.5), g1, ag.Galaxy(redshift=0.5)],
                redshift=None,
            )

            assert plane.mass_profile_phis.in_grouped_list[0][0] == pytest.approx(
                0.9, 1.0e-4
            )
            assert plane.mass_profile_phis.in_grouped_list[1][0] == pytest.approx(
                0.8, 1.0e-4
            )

            plane = ag.Plane(
                galaxies=[g0, ag.Galaxy(redshift=0.5), g1, ag.Galaxy(redshift=0.5), g2],
                redshift=None,
            )

            assert plane.mass_profile_phis.in_grouped_list[0][0] == pytest.approx(
                0.9, 1.0e-4
            )
            assert plane.mass_profile_phis.in_grouped_list[1][0] == pytest.approx(
                0.8, 1.0e-4
            )
            assert plane.mass_profile_phis.in_grouped_list[2][0] == pytest.approx(
                0.7, 1.0e-4
            )
            assert plane.mass_profile_phis.in_grouped_list[2][1] == pytest.approx(
                0.6, 1.0e-4
            )


class TestAbstractPlaneProfiles:
    class TestProfileImage:
        def test__image_from_grid__same_as_its_light_image(
            self, sub_grid_7x7, gal_x1_lp
        ):
            light_profile = gal_x1_lp.light_profiles[0]

            lp_image = light_profile.image_from_grid(grid=sub_grid_7x7)

            # Perform sub gridding average manually
            lp_image_pixel_0 = (
                lp_image[0] + lp_image[1] + lp_image[2] + lp_image[3]
            ) / 4
            lp_image_pixel_1 = (
                lp_image[4] + lp_image[5] + lp_image[6] + lp_image[7]
            ) / 4

            plane = ag.Plane(galaxies=[gal_x1_lp], redshift=None)

            image = plane.image_from_grid(grid=sub_grid_7x7)

            assert (image.slim_binned[0] == lp_image_pixel_0).all()
            assert (image.slim_binned[1] == lp_image_pixel_1).all()
            assert (image == lp_image).all()

        def test__image_from_grid__same_as_its_galaxy_image(
            self, sub_grid_7x7, gal_x1_lp
        ):
            galaxy_image = gal_x1_lp.image_from_grid(grid=sub_grid_7x7)

            plane = ag.Plane(galaxies=[gal_x1_lp], redshift=None)

            image = plane.image_from_grid(grid=sub_grid_7x7)

            assert image == pytest.approx(galaxy_image, 1.0e-4)

        def test__image_from_positions__same_as_galaxy_image_with_conversions(
            self, grid_irregular_grouped_7x7, gal_x1_lp
        ):
            galaxy_image = gal_x1_lp.image_from_grid(grid=grid_irregular_grouped_7x7)

            plane = ag.Plane(galaxies=[gal_x1_lp], redshift=None)

            image = plane.image_from_grid(grid=grid_irregular_grouped_7x7)

            assert image.in_grouped_list[0][0] == pytest.approx(
                galaxy_image.in_grouped_list[0][0], 1.0e-4
            )

        def test__images_of_galaxies(self, sub_grid_7x7):
            # Overwrite one value so intensity in each pixel is different
            sub_grid_7x7[5] = np.array([2.0, 2.0])

            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=2.0)
            )

            lp0 = g0.light_profiles[0]
            lp1 = g1.light_profiles[0]

            lp0_image = lp0.image_from_grid(grid=sub_grid_7x7)
            lp1_image = lp1.image_from_grid(grid=sub_grid_7x7)

            # Perform sub gridding average manually
            lp0_image_pixel_0 = (
                lp0_image[0] + lp0_image[1] + lp0_image[2] + lp0_image[3]
            ) / 4
            lp0_image_pixel_1 = (
                lp0_image[4] + lp0_image[5] + lp0_image[6] + lp0_image[7]
            ) / 4
            lp1_image_pixel_0 = (
                lp1_image[0] + lp1_image[1] + lp1_image[2] + lp1_image[3]
            ) / 4
            lp1_image_pixel_1 = (
                lp1_image[4] + lp1_image[5] + lp1_image[6] + lp1_image[7]
            ) / 4

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            image = plane.image_from_grid(grid=sub_grid_7x7)

            assert image.slim_binned[0] == pytest.approx(
                lp0_image_pixel_0 + lp1_image_pixel_0, 1.0e-4
            )
            assert image.slim_binned[1] == pytest.approx(
                lp0_image_pixel_1 + lp1_image_pixel_1, 1.0e-4
            )

            image_of_galaxies = plane.images_of_galaxies_from_grid(grid=sub_grid_7x7)

            assert image_of_galaxies[0].slim_binned[0] == lp0_image_pixel_0
            assert image_of_galaxies[0].slim_binned[1] == lp0_image_pixel_1
            assert image_of_galaxies[1].slim_binned[0] == lp1_image_pixel_0
            assert image_of_galaxies[1].slim_binned[1] == lp1_image_pixel_1

        def test__same_as_above__use_multiple_galaxies(self, sub_grid_7x7):
            # Overwrite one value so intensity in each pixel is different
            sub_grid_7x7[5] = np.array([2.0, 2.0])

            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=2.0)
            )

            g0_image = g0.image_from_grid(grid=sub_grid_7x7)

            g1_image = g1.image_from_grid(grid=sub_grid_7x7)

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            image = plane.image_from_grid(grid=sub_grid_7x7)

            assert image == pytest.approx(g0_image + g1_image, 1.0e-4)

        def test__same_as_above__grid_is_positions(self):
            # Overwrite one value so intensity in each pixel is different
            positions = ag.Grid2DIrregularGrouped(grid=[[(2.0, 2.0)], [(3.0, 3.0)]])

            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=2.0)
            )

            g0_image = g0.image_from_grid(grid=positions)

            g1_image = g1.image_from_grid(grid=positions)

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            image = plane.image_from_grid(grid=positions)

            assert image.in_grouped_list[0][0] == pytest.approx(
                g0_image.in_grouped_list[0][0] + g1_image.in_grouped_list[0][0], 1.0e-4
            )
            assert image.in_grouped_list[1][0] == pytest.approx(
                g0_image.in_grouped_list[1][0] + g1_image.in_grouped_list[1][0], 1.0e-4
            )

        def test__plane_has_no_galaxies__image_is_zeros_size_of_ungalaxyed_grid(
            self, sub_grid_7x7
        ):
            plane = ag.Plane(galaxies=[], redshift=0.5)

            image = plane.image_from_grid(grid=sub_grid_7x7)

            assert image.shape_native == (7, 7)
            assert (image[0] == 0.0).all()
            assert (image[1] == 0.0).all()

        def test__x1_plane__padded_image__compare_to_galaxy_images_using_padded_grid_stack(
            self, sub_grid_7x7
        ):

            padded_grid = sub_grid_7x7.padded_grid_from_kernel_shape(
                kernel_shape_native=(3, 3)
            )

            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=2.0)
            )
            g2 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=3.0)
            )

            padded_g0_image = g0.image_from_grid(grid=padded_grid)

            padded_g1_image = g1.image_from_grid(grid=padded_grid)

            padded_g2_image = g2.image_from_grid(grid=padded_grid)

            plane = ag.Plane(galaxies=[g0, g1, g2])

            padded_plane_image = plane.padded_image_from_grid_and_psf_shape(
                grid=sub_grid_7x7, psf_shape_2d=(3, 3)
            )

            assert padded_plane_image.shape_native == (9, 9)
            assert padded_plane_image == pytest.approx(
                padded_g0_image + padded_g1_image + padded_g2_image, 1.0e-4
            )

        def test__galaxy_image_dict_from_grid(self, sub_grid_7x7):

            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
                light_profile=ag.lp.EllipticalSersic(intensity=2.0),
            )

            g2 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=3.0)
            )

            g0_image = g0.image_from_grid(grid=sub_grid_7x7)
            g1_image = g1.image_from_grid(grid=sub_grid_7x7)
            g2_image = g2.image_from_grid(grid=sub_grid_7x7)

            plane = ag.Plane(redshift=-0.75, galaxies=[g1, g0, g2])

            image_1d_dict = plane.galaxy_image_dict_from_grid(grid=sub_grid_7x7)

            assert (image_1d_dict[g0].slim == g0_image).all()
            assert (image_1d_dict[g1].slim == g1_image).all()
            assert (image_1d_dict[g2].slim == g2_image).all()

            image_dict = plane.galaxy_image_dict_from_grid(grid=sub_grid_7x7)

            assert (image_dict[g0].native == g0_image.native).all()
            assert (image_dict[g1].native == g1_image.native).all()
            assert (image_dict[g2].native == g2_image.native).all()

    class TestConvergence:
        def test__convergence_same_as_multiple_galaxies__include_reshape_mapping(
            self, sub_grid_7x7
        ):
            # The *ungalaxyed* sub-grid must be used to compute the convergence. This changes the subgrid to ensure this
            # is the case.

            sub_grid_7x7[5] = np.array([5.0, 2.0])

            g0 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(
                    einstein_radius=1.0, centre=(1.0, 0.0)
                ),
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(
                    einstein_radius=2.0, centre=(1.0, 1.0)
                ),
            )

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_convergence = mp0.convergence_from_grid(grid=sub_grid_7x7)
            mp1_sub_convergence = mp1.convergence_from_grid(grid=sub_grid_7x7)

            mp_sub_convergence = mp0_sub_convergence + mp1_sub_convergence

            # Perform sub gridding average manually

            mp_convergence_pixel_0 = (
                mp_sub_convergence[0]
                + mp_sub_convergence[1]
                + mp_sub_convergence[2]
                + mp_sub_convergence[3]
            ) / 4
            mp_convergence_pixel_1 = (
                mp_sub_convergence[4]
                + mp_sub_convergence[5]
                + mp_sub_convergence[6]
                + mp_sub_convergence[7]
            ) / 4

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            convergence = plane.convergence_from_grid(grid=sub_grid_7x7)

            assert convergence.native_binned[2, 2] == pytest.approx(
                mp_convergence_pixel_0, 1.0e-4
            )
            assert convergence.native_binned[2, 3] == pytest.approx(
                mp_convergence_pixel_1, 1.0e-4
            )

        def test__same_as_above_galaxies___use_galaxy_to_compute_convergence(
            self, sub_grid_7x7
        ):
            g0 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=2.0),
            )

            g0_convergence = g0.convergence_from_grid(grid=sub_grid_7x7)

            g1_convergence = g1.convergence_from_grid(grid=sub_grid_7x7)

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            convergence = plane.convergence_from_grid(grid=sub_grid_7x7)

            assert convergence == pytest.approx(g0_convergence + g1_convergence, 1.0e-8)

        def test__convergence_from_grid_as_positions(self, grid_irregular_grouped_7x7):
            g0 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
            )

            g0_convergence = g0.convergence_from_grid(grid=grid_irregular_grouped_7x7)

            plane = ag.Plane(galaxies=[g0], redshift=None)

            convergence = plane.convergence_from_grid(grid=grid_irregular_grouped_7x7)

            assert convergence.in_grouped_list[0][0] == pytest.approx(
                g0_convergence.in_grouped_list[0][0], 1.0e-8
            )

        def test__plane_has_no_galaxies__convergence_is_zeros_size_of_reshaped_sub_array(
            self, sub_grid_7x7
        ):
            plane = ag.Plane(galaxies=[], redshift=0.5)

            convergence = plane.convergence_from_grid(grid=sub_grid_7x7)

            assert convergence.sub_shape_slim == sub_grid_7x7.sub_shape_slim

            convergence = plane.convergence_from_grid(grid=sub_grid_7x7)

            assert convergence.sub_shape_native == (14, 14)

            convergence = plane.convergence_from_grid(grid=sub_grid_7x7)

            assert convergence.shape_native == (7, 7)

    class TestPotential:
        def test__potential_same_as_multiple_galaxies__include_reshape_mapping(
            self, sub_grid_7x7
        ):
            # The *ungalaxyed* sub-grid must be used to compute the potential. This changes the subgrid to ensure this
            # is the case.

            sub_grid_7x7[5] = np.array([5.0, 2.0])

            g0 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(
                    einstein_radius=1.0, centre=(1.0, 0.0)
                ),
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(
                    einstein_radius=2.0, centre=(1.0, 1.0)
                ),
            )

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_sub_potential = mp0.potential_from_grid(grid=sub_grid_7x7)
            mp1_sub_potential = mp1.potential_from_grid(grid=sub_grid_7x7)

            mp_sub_potential = mp0_sub_potential + mp1_sub_potential

            # Perform sub gridding average manually

            mp_potential_pixel_0 = (
                mp_sub_potential[0]
                + mp_sub_potential[1]
                + mp_sub_potential[2]
                + mp_sub_potential[3]
            ) / 4
            mp_potential_pixel_1 = (
                mp_sub_potential[4]
                + mp_sub_potential[5]
                + mp_sub_potential[6]
                + mp_sub_potential[7]
            ) / 4

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            potential = plane.potential_from_grid(grid=sub_grid_7x7)

            assert potential.native_binned[2, 2] == pytest.approx(
                mp_potential_pixel_0, 1.0e-4
            )
            assert potential.native_binned[2, 3] == pytest.approx(
                mp_potential_pixel_1, 1.0e-4
            )

        def test__same_as_above_galaxies___use_galaxy_to_compute_potential(
            self, sub_grid_7x7
        ):
            g0 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=2.0),
            )

            g0_potential = g0.potential_from_grid(grid=sub_grid_7x7)

            g1_potential = g1.potential_from_grid(grid=sub_grid_7x7)

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            potential = plane.potential_from_grid(grid=sub_grid_7x7)

            assert potential == pytest.approx(g0_potential + g1_potential, 1.0e-8)

        def test__potential_from_grid_as_positions(self, grid_irregular_grouped_7x7):
            g0 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
            )

            print(grid_irregular_grouped_7x7)

            g0_potential = g0.potential_from_grid(grid=grid_irregular_grouped_7x7)

            plane = ag.Plane(galaxies=[g0], redshift=None)

            potential = plane.potential_from_grid(grid=grid_irregular_grouped_7x7)

            assert potential.in_grouped_list[0][0] == pytest.approx(
                g0_potential.in_grouped_list[0][0], 1.0e-8
            )

        def test__plane_has_no_galaxies__potential_is_zeros_size_of_reshaped_sub_array(
            self, sub_grid_7x7
        ):
            plane = ag.Plane(galaxies=[], redshift=0.5)

            potential = plane.potential_from_grid(grid=sub_grid_7x7)

            assert potential.sub_shape_slim == sub_grid_7x7.sub_shape_slim

            potential = plane.potential_from_grid(grid=sub_grid_7x7)

            assert potential.sub_shape_native == (14, 14)

            potential = plane.potential_from_grid(grid=sub_grid_7x7)

            assert potential.shape_native == (7, 7)

    class TestDeflections:
        def test__deflections_from_plane__same_as_the_galaxy_mass_profiles(
            self, sub_grid_7x7
        ):
            # Overwrite one value so intensity in each pixel is different
            sub_grid_7x7[5] = np.array([2.0, 2.0])

            g0 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=2.0),
            )

            mp0 = g0.mass_profiles[0]
            mp1 = g1.mass_profiles[0]

            mp0_image = mp0.deflections_from_grid(grid=sub_grid_7x7)
            mp1_image = mp1.deflections_from_grid(grid=sub_grid_7x7)

            # Perform sub gridding average manually
            mp0_image_pixel_0x = (
                mp0_image[0, 0] + mp0_image[1, 0] + mp0_image[2, 0] + mp0_image[3, 0]
            ) / 4
            mp0_image_pixel_1x = (
                mp0_image[4, 0] + mp0_image[5, 0] + mp0_image[6, 0] + mp0_image[7, 0]
            ) / 4
            mp0_image_pixel_0y = (
                mp0_image[0, 1] + mp0_image[1, 1] + mp0_image[2, 1] + mp0_image[3, 1]
            ) / 4
            mp0_image_pixel_1y = (
                mp0_image[4, 1] + mp0_image[5, 1] + mp0_image[6, 1] + mp0_image[7, 1]
            ) / 4

            mp1_image_pixel_0x = (
                mp1_image[0, 0] + mp1_image[1, 0] + mp1_image[2, 0] + mp1_image[3, 0]
            ) / 4
            mp1_image_pixel_1x = (
                mp1_image[4, 0] + mp1_image[5, 0] + mp1_image[6, 0] + mp1_image[7, 0]
            ) / 4
            mp1_image_pixel_0y = (
                mp1_image[0, 1] + mp1_image[1, 1] + mp1_image[2, 1] + mp1_image[3, 1]
            ) / 4
            mp1_image_pixel_1y = (
                mp1_image[4, 1] + mp1_image[5, 1] + mp1_image[6, 1] + mp1_image[7, 1]
            ) / 4

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            deflections = plane.deflections_from_grid(grid=sub_grid_7x7)

            assert deflections.slim_binned[0, 0] == pytest.approx(
                mp0_image_pixel_0x + mp1_image_pixel_0x, 1.0e-4
            )
            assert deflections.slim_binned[1, 0] == pytest.approx(
                mp0_image_pixel_1x + mp1_image_pixel_1x, 1.0e-4
            )
            assert deflections.slim_binned[0, 1] == pytest.approx(
                mp0_image_pixel_0y + mp1_image_pixel_0y, 1.0e-4
            )
            assert deflections.slim_binned[1, 1] == pytest.approx(
                mp0_image_pixel_1y + mp1_image_pixel_1y, 1.0e-4
            )

        def test__deflections_same_as_its_galaxy___use_multiple_galaxies(
            self, sub_grid_7x7
        ):
            # Overwrite one value so intensity in each pixel is different
            sub_grid_7x7[5] = np.array([2.0, 2.0])

            g0 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=2.0),
            )

            g0_deflections = g0.deflections_from_grid(grid=sub_grid_7x7)

            g1_deflections = g1.deflections_from_grid(grid=sub_grid_7x7)

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            deflections = plane.deflections_from_grid(grid=sub_grid_7x7)

            assert deflections == pytest.approx(g0_deflections + g1_deflections, 1.0e-4)

        def test__deflections_from_grid_as_positions(self, grid_irregular_grouped_7x7):
            g0 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
            )

            g0_deflections = g0.deflections_from_grid(grid=grid_irregular_grouped_7x7)

            plane = ag.Plane(galaxies=[g0], redshift=None)

            deflections = plane.deflections_from_grid(grid=grid_irregular_grouped_7x7)

            assert deflections.in_grouped_list[0][0][0] == pytest.approx(
                g0_deflections.in_grouped_list[0][0][0], 1.0e-8
            )
            assert deflections.in_grouped_list[0][0][1] == pytest.approx(
                g0_deflections.in_grouped_list[0][0][1], 1.0e-8
            )

        def test__deflections_numerics__x2_galaxy_in_plane__or_galaxy_x2_sis__deflections_double(
            self, grid_7x7, gal_x1_mp, gal_x2_mp
        ):
            plane = ag.Plane(galaxies=[gal_x2_mp], redshift=None)

            deflections = plane.deflections_from_grid(grid=grid_7x7)

            assert deflections[0:2] == pytest.approx(
                np.array([[3.0 * 0.707, -3.0 * 0.707], [3.0, 0.0]]), 1e-3
            )

            plane = ag.Plane(galaxies=[gal_x1_mp, gal_x1_mp], redshift=None)

            deflections = plane.deflections_from_grid(grid=grid_7x7)

            assert deflections[0:2] == pytest.approx(
                np.array([[2.0 * 0.707, -2.0 * 0.707], [2.0, 0.0]]), 1e-3
            )

        def test__plane_has_no_galaxies__deflections_are_zeros_size_of_ungalaxyed_grid(
            self, sub_grid_7x7
        ):
            plane = ag.Plane(redshift=0.5, galaxies=[])

            deflections = plane.deflections_from_grid(grid=sub_grid_7x7)

            assert deflections.shape_native == (7, 7)
            assert (deflections.slim_binned[0, 0] == 0.0).all()
            assert (deflections.slim_binned[0, 1] == 0.0).all()
            assert (deflections.slim_binned[1, 0] == 0.0).all()
            assert (deflections.slim_binned[0] == 0.0).all()

    class TestLensingObject:
        def test__correct_einstein_mass_caclulated_for_multiple_mass_profiles__means_all_innherited_methods_work(
            self,
        ):

            grid = ag.Grid2D.uniform(shape_native=(50, 50), pixel_scales=0.15)

            sis_0 = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.2)

            sis_1 = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.4)

            sis_2 = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.6)

            sis_3 = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=0.8)

            galaxy_0 = ag.Galaxy(
                mass_profile_0=sis_0, mass_profile_1=sis_1, redshift=0.5
            )
            galaxy_1 = ag.Galaxy(
                mass_profile_0=sis_2, mass_profile_1=sis_3, redshift=0.5
            )

            plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

            einstein_mass = plane.einstein_mass_angular_from_grid(grid=grid)

            assert einstein_mass == pytest.approx(np.pi * 2.0 ** 2.0, 1.0e-1)


class TestAbstractPlaneData:
    class TestBlurredImagePlaneImage:
        def test__blurred_image_from_grid_and_psf(
            self, sub_grid_7x7, blurring_grid_7x7, psf_3x3, convolver_7x7
        ):

            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=1.0, light_profile=ag.lp.EllipticalSersic(intensity=2.0)
            )

            blurred_g0_image = g0.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                blurring_grid=blurring_grid_7x7,
                convolver=convolver_7x7,
            )

            blurred_g1_image = g1.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                blurring_grid=blurring_grid_7x7,
                convolver=convolver_7x7,
            )

            plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

            blurred_image = plane.blurred_image_from_grid_and_psf(
                grid=sub_grid_7x7, blurring_grid=blurring_grid_7x7, psf=psf_3x3
            )

            assert blurred_image.slim == pytest.approx(
                blurred_g0_image.slim + blurred_g1_image.slim, 1.0e-4
            )

            assert blurred_image.native == pytest.approx(
                blurred_g0_image.native + blurred_g1_image.native, 1.0e-4
            )

        def test__blurred_image_of_galaxies_from_grid_and_psf(
            self, sub_grid_7x7, blurring_grid_7x7, psf_3x3
        ):
            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=1.0, light_profile=ag.lp.EllipticalSersic(intensity=2.0)
            )

            blurred_g0_image = g0.blurred_image_from_grid_and_psf(
                grid=sub_grid_7x7, blurring_grid=blurring_grid_7x7, psf=psf_3x3
            )

            blurred_g1_image = g1.blurred_image_from_grid_and_psf(
                grid=sub_grid_7x7, blurring_grid=blurring_grid_7x7, psf=psf_3x3
            )

            plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

            blurred_images_of_galaxies = plane.blurred_images_of_galaxies_from_grid_and_psf(
                grid=sub_grid_7x7, blurring_grid=blurring_grid_7x7, psf=psf_3x3
            )

            assert blurred_g0_image.shape_slim == 9
            assert blurred_images_of_galaxies[0].slim == pytest.approx(
                blurred_g0_image.slim, 1.0e-4
            )
            assert blurred_g1_image.shape_slim == 9
            assert blurred_images_of_galaxies[1].slim == pytest.approx(
                blurred_g1_image.slim, 1.0e-4
            )

            assert blurred_images_of_galaxies[0].native == pytest.approx(
                blurred_g0_image.native, 1.0e-4
            )
            assert blurred_images_of_galaxies[1].native == pytest.approx(
                blurred_g1_image.native, 1.0e-4
            )

        def test__blurred_image_from_grid_and_convolver(
            self, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
        ):
            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=1.0, light_profile=ag.lp.EllipticalSersic(intensity=2.0)
            )

            blurred_g0_image = g0.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            blurred_g1_image = g1.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

            blurred_image = plane.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            assert blurred_image.slim == pytest.approx(
                blurred_g0_image.slim + blurred_g1_image.slim, 1.0e-4
            )

            assert blurred_image.native == pytest.approx(
                blurred_g0_image.native + blurred_g1_image.native, 1.0e-4
            )

        def test__blurred_image_of_galaxies_from_grid_and_convolver(
            self, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
        ):
            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=1.0, light_profile=ag.lp.EllipticalSersic(intensity=2.0)
            )

            blurred_g0_image = g0.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            blurred_g1_image = g1.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

            blurred_images_of_galaxies = plane.blurred_images_of_galaxies_from_grid_and_convolver(
                grid=sub_grid_7x7,
                blurring_grid=blurring_grid_7x7,
                convolver=convolver_7x7,
            )

            assert blurred_g0_image.shape_slim == 9
            assert blurred_images_of_galaxies[0].slim == pytest.approx(
                blurred_g0_image.slim, 1.0e-4
            )
            assert blurred_g1_image.shape_slim == 9
            assert blurred_images_of_galaxies[1].slim == pytest.approx(
                blurred_g1_image.slim, 1.0e-4
            )

            assert blurred_images_of_galaxies[0].native == pytest.approx(
                blurred_g0_image.native, 1.0e-4
            )
            assert blurred_images_of_galaxies[1].native == pytest.approx(
                blurred_g1_image.native, 1.0e-4
            )

        def test__galaxy_blurred_image_dict_from_grid_and_convolver(
            self, sub_grid_7x7, blurring_grid_7x7, convolver_7x7
        ):

            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
                light_profile=ag.lp.EllipticalSersic(intensity=2.0),
            )

            g2 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=3.0)
            )

            g0_blurred_image = g0.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            g1_blurred_image = g1.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            g2_blurred_image = g2.blurred_image_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            plane = ag.Plane(redshift=-0.75, galaxies=[g1, g0, g2])

            blurred_image_dict = plane.galaxy_blurred_image_dict_from_grid_and_convolver(
                grid=sub_grid_7x7,
                convolver=convolver_7x7,
                blurring_grid=blurring_grid_7x7,
            )

            assert (blurred_image_dict[g0].slim == g0_blurred_image.slim).all()
            assert (blurred_image_dict[g1].slim == g1_blurred_image.slim).all()
            assert (blurred_image_dict[g2].slim == g2_blurred_image.slim).all()

    class TestUnmaskedBlurredProfileImages:
        def test__unmasked_images_of_plane_planes_and_galaxies(self):
            psf = ag.Kernel2D.manual_native(
                array=(np.array([[0.0, 3.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]])),
                pixel_scales=1.0,
            )

            mask = ag.Mask2D.manual(
                mask=[[True, True, True], [True, False, True], [True, True, True]],
                pixel_scales=1.0,
                sub_size=1,
            )

            grid = ag.Grid2D.from_mask(mask=mask)

            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=0.1)
            )
            g1 = ag.Galaxy(
                redshift=1.0, light_profile=ag.lp.EllipticalSersic(intensity=0.2)
            )

            plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

            padded_grid = grid.padded_grid_from_kernel_shape(
                kernel_shape_native=psf.shape_native
            )

            manual_blurred_image_0 = plane.images_of_galaxies_from_grid(
                grid=padded_grid
            )[0]
            manual_blurred_image_0 = psf.convolved_array_from_array(
                array=manual_blurred_image_0
            )

            manual_blurred_image_1 = plane.images_of_galaxies_from_grid(
                grid=padded_grid
            )[1]
            manual_blurred_image_1 = psf.convolved_array_from_array(
                array=manual_blurred_image_1
            )

            unmasked_blurred_image = plane.unmasked_blurred_image_from_grid_and_psf(
                grid=grid, psf=psf
            )

            assert unmasked_blurred_image.native == pytest.approx(
                manual_blurred_image_0.native_binned[1:4, 1:4]
                + manual_blurred_image_1.native_binned[1:4, 1:4],
                1.0e-4,
            )

            unmasked_blurred_image_of_galaxies = plane.unmasked_blurred_image_of_galaxies_from_grid_and_psf(
                grid=grid, psf=psf
            )

            assert unmasked_blurred_image_of_galaxies[0].native == pytest.approx(
                manual_blurred_image_0.native_binned[1:4, 1:4], 1.0e-4
            )

            assert unmasked_blurred_image_of_galaxies[1].native == pytest.approx(
                manual_blurred_image_1.native_binned[1:4, 1:4], 1.0e-4
            )

    class TestVisibilities:
        def test__visibilities_from_grid_and_transformer(
            self, sub_grid_7x7, transformer_7x7_7
        ):
            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )

            image = g0.image_from_grid(grid=sub_grid_7x7)

            visibilities = transformer_7x7_7.visibilities_from_image(image=image)

            plane = ag.Plane(redshift=0.5, galaxies=[g0])

            plane_visibilities = plane.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            assert (visibilities == plane_visibilities).all()

            g1 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=2.0)
            )

            image = g0.image_from_grid(grid=sub_grid_7x7) + g1.image_from_grid(
                grid=sub_grid_7x7
            )

            visibilities = transformer_7x7_7.visibilities_from_image(image=image)

            plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

            plane_visibilities = plane.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            assert visibilities == pytest.approx(plane_visibilities, 1.0e-4)

        def test__visibilities_from_grid_and_transformer__plane_has_no_galaxies__returns_zeros(
            self, sub_grid_7x7, transformer_7x7_7
        ):
            plane = ag.Plane(redshift=0.5, galaxies=[])

            plane_visibilities = plane.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            assert (plane_visibilities.slim == 0.0 + 0.0j * np.zeros((7,))).all()

        def test__visibilities_of_galaxies_from_grid_and_transformer(
            self, sub_grid_7x7, transformer_7x7_7
        ):
            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )

            g1 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=2.0)
            )

            g0_image = g0.image_from_grid(grid=sub_grid_7x7)

            g1_image = g1.image_from_grid(grid=sub_grid_7x7)

            g0_visibilities = transformer_7x7_7.visibilities_from_image(image=g0_image)

            g1_visibilities = transformer_7x7_7.visibilities_from_image(image=g1_image)

            plane = ag.Plane(redshift=0.5, galaxies=[g0, g1])

            plane_visibilities_of_galaxies = plane.profile_visibilities_of_galaxies_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            assert (g0_visibilities == plane_visibilities_of_galaxies[0]).all()
            assert (g1_visibilities == plane_visibilities_of_galaxies[1]).all()

            plane_visibilities = plane.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            assert sum(plane_visibilities_of_galaxies) == pytest.approx(
                plane_visibilities, 1.0e-4
            )

        def test__galaxy_visibilities_dict_from_grid_and_transformer(
            self, sub_grid_7x7, transformer_7x7_7
        ):

            g0 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=1.0)
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
                light_profile=ag.lp.EllipticalSersic(intensity=2.0),
            )

            g2 = ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.EllipticalSersic(intensity=3.0)
            )

            g3 = ag.Galaxy(
                redshift=1.0, light_profile=ag.lp.EllipticalSersic(intensity=5.0)
            )

            g0_visibilities = g0.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            g1_visibilities = g1.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            g2_visibilities = g2.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            plane = ag.Plane(redshift=-0.75, galaxies=[g1, g0, g2])

            visibilities_dict = plane.galaxy_profile_visibilities_dict_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            assert (visibilities_dict[g0] == g0_visibilities).all()
            assert (visibilities_dict[g1] == g1_visibilities).all()
            assert (visibilities_dict[g2] == g2_visibilities).all()

    class TestGrid2DIrregular:
        def test__no_galaxies_with_pixelizations_in_plane__returns_none(
            self, sub_grid_7x7
        ):
            galaxy_no_pix = ag.Galaxy(redshift=0.5)

            plane = ag.Plane(galaxies=[galaxy_no_pix], redshift=0.5)

            sparse_grid = plane.sparse_image_plane_grid_from_grid(grid=sub_grid_7x7)

            assert sparse_grid is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_sparse_grid(
            self, sub_grid_7x7
        ):
            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=1, grid=[[1.0, 1.0]]),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )

            plane = ag.Plane(galaxies=[galaxy_pix], redshift=0.5)

            sparse_grid = plane.sparse_image_plane_grid_from_grid(grid=sub_grid_7x7)

            assert (sparse_grid == np.array([[1.0, 1.0]])).all()

        def test__1_galaxy_in_plane__it_has_pixelization_and_hyper_image_returns_sparse_grid_and_uses_hyper_image(
            self, sub_grid_7x7
        ):
            # In the MockPixelization class the grid is returned if hyper image=None, and grid*hyper image is
            # returned otherwise.

            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(
                    value=1, grid=np.array([[1.0, 1.0]])
                ),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
                hyper_galaxy_image=2,
            )

            plane = ag.Plane(galaxies=[galaxy_pix], redshift=0.5)

            sparse_grid = plane.sparse_image_plane_grid_from_grid(grid=sub_grid_7x7)

            assert (sparse_grid == np.array([[2.0, 2.0]])).all()

    class TestMapper:
        def test__no_galaxies_with_pixelizations_in_plane__returns_none(
            self, sub_grid_7x7
        ):
            galaxy_no_pix = ag.Galaxy(redshift=0.5)

            plane = ag.Plane(galaxies=[galaxy_no_pix], redshift=0.5)

            mapper = plane.mapper_from_grid_and_sparse_grid(
                grid=sub_grid_7x7, sparse_grid=sub_grid_7x7
            )

            assert mapper is None

        def test__1_galaxy_in_plane__it_has_pixelization__returns_mapper(
            self, sub_grid_7x7
        ):
            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=1),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )

            plane = ag.Plane(galaxies=[galaxy_pix], redshift=0.5)

            mapper = plane.mapper_from_grid_and_sparse_grid(
                grid=sub_grid_7x7, sparse_grid=sub_grid_7x7
            )

            assert mapper == 1

            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=1),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_no_pix = ag.Galaxy(redshift=0.5)

            plane = ag.Plane(galaxies=[galaxy_no_pix, galaxy_pix], redshift=0.5)

            mapper = plane.mapper_from_grid_and_sparse_grid(
                grid=sub_grid_7x7, sparse_grid=sub_grid_7x7
            )

            assert mapper == 1

        def test__inversion_use_border_is_false__still_returns_mapper(
            self, sub_grid_7x7
        ):
            galaxy_pix = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=1),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_no_pix = ag.Galaxy(redshift=0.5)

            plane = ag.Plane(galaxies=[galaxy_no_pix, galaxy_pix], redshift=0.5)

            mapper = plane.mapper_from_grid_and_sparse_grid(
                grid=sub_grid_7x7,
                sparse_grid=sub_grid_7x7,
                settings_pixelization=ag.SettingsPixelization(use_border=False),
            )

            assert mapper == 1

        def test__2_galaxies_in_plane__both_have_pixelization__raises_error(
            self, sub_grid_7x7
        ):
            galaxy_pix_0 = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=1),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )
            galaxy_pix_1 = ag.Galaxy(
                redshift=0.5,
                pixelization=mock.MockPixelization(value=2),
                regularization=mock.MockRegularization(matrix_shape=(1, 1)),
            )

            plane = ag.Plane(galaxies=[galaxy_pix_0, galaxy_pix_1], redshift=None)

            with pytest.raises(exc.PixelizationException):
                plane.mapper_from_grid_and_sparse_grid(
                    grid=sub_grid_7x7,
                    sparse_grid=sub_grid_7x7,
                    settings_pixelization=ag.SettingsPixelization(use_border=False),
                )

    class TestInversion:
        def test__x1_inversion_imaging_in_plane__performs_inversion_correctly(
            self, sub_grid_7x7, masked_imaging_7x7
        ):

            pix = ag.pix.Rectangular(shape=(3, 3))
            reg = ag.reg.Constant(coefficient=0.0)

            g0 = ag.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

            inversion = plane.inversion_imaging_from_grid_and_data(
                grid=sub_grid_7x7,
                image=masked_imaging_7x7.image,
                noise_map=masked_imaging_7x7.noise_map,
                convolver=masked_imaging_7x7.convolver,
                settings_pixelization=ag.SettingsPixelization(use_border=False),
            )

            assert inversion.mapped_reconstructed_image == pytest.approx(
                masked_imaging_7x7.image, 1.0e-2
            )

        def test__x1_inversion_interferometer_in_plane__performs_inversion_correctly(
            self, sub_grid_7x7, masked_interferometer_7
        ):

            masked_interferometer_7.visibilities = ag.Visibilities.ones(shape_slim=(7,))

            pix = ag.pix.Rectangular(shape=(7, 7))
            reg = ag.reg.Constant(coefficient=0.0)

            g0 = ag.Galaxy(redshift=0.5, pixelization=pix, regularization=reg)

            plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5), g0])

            inversion = plane.inversion_interferometer_from_grid_and_data(
                grid=sub_grid_7x7,
                visibilities=masked_interferometer_7.visibilities,
                noise_map=masked_interferometer_7.noise_map,
                transformer=masked_interferometer_7.transformer,
                settings_pixelization=ag.SettingsPixelization(use_border=False),
                settings_inversion=ag.SettingsInversion(use_linear_operators=False),
            )

            assert inversion.mapped_reconstructed_visibilities.real == pytest.approx(
                masked_interferometer_7.visibilities.real, 1.0e-2
            )

    class TestPlaneImage:
        def test__3x3_grid__extracts_max_min_coordinates__ignores_other_coordinates_more_central(
            self, sub_grid_7x7
        ):
            sub_grid_7x7[1] = np.array([2.0, 2.0])

            galaxy = ag.Galaxy(
                redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0)
            )

            plane = ag.Plane(galaxies=[galaxy], redshift=None)

            plane_image_from_func = ag.plane.plane.plane_util.plane_image_of_galaxies_from(
                shape=(7, 7),
                grid=sub_grid_7x7.mask.unmasked_grid_sub_1,
                galaxies=[galaxy],
            )

            plane_image_from_plane = plane.plane_image_from_grid(grid=sub_grid_7x7)

            assert (plane_image_from_func.array == plane_image_from_plane.array).all()

        def test__ensure_index_of_plane_image_has_negative_arcseconds_at_start(self,):
            # The grid coordinates -2.0 -> 2.0 mean a plane of shape (5,5) has arc second coordinates running over
            # -1.6, -0.8, 0.0, 0.8, 1.6. The origin -1.6, -1.6 of the model_galaxy means its brighest pixel should be
            # index 0 of the 1D grid and (0,0) of the 2d plane datas_.

            mask = ag.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0, sub_size=1)

            grid = ag.Grid2D.from_mask(mask=mask)

            g0 = ag.Galaxy(
                redshift=0.5,
                light_profile=ag.lp.EllipticalSersic(centre=(1.6, -1.6), intensity=1.0),
            )
            plane = ag.Plane(galaxies=[g0], redshift=None)

            plane_image = plane.plane_image_from_grid(grid=grid)

            assert plane_image.array.shape_native == (5, 5)
            assert np.unravel_index(
                plane_image.array.native.argmax(), plane_image.array.native.shape
            ) == (0, 0)

            g0 = ag.Galaxy(
                redshift=0.5,
                light_profile=ag.lp.EllipticalSersic(centre=(1.6, 1.6), intensity=1.0),
            )
            plane = ag.Plane(galaxies=[g0], redshift=None)

            plane_image = plane.plane_image_from_grid(grid=grid)

            assert np.unravel_index(
                plane_image.array.native.argmax(), plane_image.array.native.shape
            ) == (0, 4)

            g0 = ag.Galaxy(
                redshift=0.5,
                light_profile=ag.lp.EllipticalSersic(
                    centre=(-1.6, -1.6), intensity=1.0
                ),
            )
            plane = ag.Plane(galaxies=[g0], redshift=None)

            plane_image = plane.plane_image_from_grid(grid=grid)

            assert np.unravel_index(
                plane_image.array.native.argmax(), plane_image.array.native.shape
            ) == (4, 0)

            g0 = ag.Galaxy(
                redshift=0.5,
                light_profile=ag.lp.EllipticalSersic(centre=(-1.6, 1.6), intensity=1.0),
            )
            plane = ag.Plane(galaxies=[g0], redshift=None)

            plane_image = plane.plane_image_from_grid(grid=grid)

            assert np.unravel_index(
                plane_image.array.native.argmax(), plane_image.array.native.shape
            ) == (4, 4)

    class TestContributionMaps:
        def test__x2_hyper_galaxy__use_numerical_values_for_noise_scaling(self):
            hyper_galaxy_0 = ag.HyperGalaxy(
                contribution_factor=0.0, noise_factor=0.0, noise_power=1.0
            )
            hyper_galaxy_1 = ag.HyperGalaxy(
                contribution_factor=1.0, noise_factor=0.0, noise_power=1.0
            )

            hyper_model_image = ag.Array2D.manual_native(
                array=[[0.5, 1.0, 1.5]], pixel_scales=1.0
            )

            hyper_galaxy_image_0 = ag.Array2D.manual_native(
                array=[[0.5, 1.0, 1.5]], pixel_scales=1.0
            )
            hyper_galaxy_image_1 = ag.Array2D.manual_native(
                array=[[0.5, 1.0, 1.5]], pixel_scales=1.0
            )

            galaxy_0 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_0,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image_0,
            )

            galaxy_1 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_1,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image_1,
            )

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0, galaxy_1])

            assert (
                plane.contribution_maps_of_galaxies[0].native
                == np.array([[1.0, 1.0, 1.0]])
            ).all()
            assert (
                plane.contribution_maps_of_galaxies[1].native
                == np.array([[5.0 / 9.0, (1.0 / 2.0) / (1.5 / 2.5), 1.0]])
            ).all()

        def test__contribution_maps_are_same_as_hyper_galaxy_calculation(self):
            hyper_model_image = ag.Array2D.manual_native(
                [[2.0, 4.0, 10.0]], pixel_scales=1.0
            )
            hyper_galaxy_image = ag.Array2D.manual_native(
                [[1.0, 5.0, 8.0]], pixel_scales=1.0
            )

            hyper_galaxy_0 = ag.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = ag.HyperGalaxy(contribution_factor=10.0)

            contribution_map_0 = hyper_galaxy_0.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            contribution_map_1 = hyper_galaxy_1.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            galaxy_0 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_0,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            galaxy_1 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_1,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0])

            assert (
                plane.contribution_maps_of_galaxies[0].slim == contribution_map_0
            ).all()

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_1])

            assert (
                plane.contribution_maps_of_galaxies[0].slim == contribution_map_1
            ).all()

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_1, galaxy_0])

            assert (
                plane.contribution_maps_of_galaxies[0].slim == contribution_map_1
            ).all()
            assert (
                plane.contribution_maps_of_galaxies[1].slim == contribution_map_0
            ).all()

        def test__contriution_maps_are_none_for_galaxy_without_hyper_galaxy(self):
            hyper_model_image = ag.Array2D.manual_native(
                [[2.0, 4.0, 10.0]], pixel_scales=1.0
            )
            hyper_galaxy_image = ag.Array2D.manual_native(
                [[1.0, 5.0, 8.0]], pixel_scales=1.0
            )

            hyper_galaxy = ag.HyperGalaxy(contribution_factor=5.0)

            contribution_map = hyper_galaxy.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            galaxy = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            plane = ag.Plane(
                redshift=0.5,
                galaxies=[galaxy, ag.Galaxy(redshift=0.5), ag.Galaxy(redshift=0.5)],
            )

            assert (
                plane.contribution_maps_of_galaxies[0].slim == contribution_map
            ).all()
            assert plane.contribution_maps_of_galaxies[1] == None
            assert plane.contribution_maps_of_galaxies[2] == None

        def test__contribution_map_is_sum_of_galaxy_contribution_maps__handles_nones_correctly(
            self,
        ):
            hyper_galaxy_0 = ag.HyperGalaxy(
                contribution_factor=0.0, noise_factor=0.0, noise_power=1.0
            )
            hyper_galaxy_1 = ag.HyperGalaxy(
                contribution_factor=1.0, noise_factor=0.0, noise_power=1.0
            )

            hyper_model_image = ag.Array2D.manual_native(
                array=[[0.5, 1.0, 1.5]], pixel_scales=1.0
            )

            hyper_galaxy_image_0 = ag.Array2D.manual_native(
                array=[[0.5, 1.0, 1.5]], pixel_scales=1.0
            )
            hyper_galaxy_image_1 = ag.Array2D.manual_native(
                array=[[0.5, 1.0, 1.5]], pixel_scales=1.0
            )

            galaxy_0 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_0,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image_0,
            )

            galaxy_1 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_1,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image_1,
            )

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0, galaxy_1])

            assert (
                sum(plane.contribution_maps_of_galaxies) == plane.contribution_map
            ).all()

            galaxy_1 = ag.Galaxy(redshift=0.5)

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0, galaxy_1])

            assert (galaxy_0.contribution_map == plane.contribution_map).all()

            galaxy_0 = ag.Galaxy(redshift=0.5)

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0, galaxy_1])

            assert plane.contribution_map == None

    class TestHyperNoiseMap:
        def test__x2_hyper_galaxy__use_numerical_values_of_hyper_noise_map_scaling(
            self,
        ):
            noise_map = ag.Array2D.manual_native(
                array=[[1.0, 2.0, 3.0]], pixel_scales=1.0
            )

            hyper_galaxy_0 = ag.HyperGalaxy(
                contribution_factor=0.0, noise_factor=1.0, noise_power=1.0
            )
            hyper_galaxy_1 = ag.HyperGalaxy(
                contribution_factor=3.0, noise_factor=1.0, noise_power=2.0
            )

            hyper_model_image = ag.Array2D.manual_native(
                array=[[0.5, 1.0, 1.5]], pixel_scales=1.0
            )

            hyper_galaxy_image_0 = ag.Array2D.manual_native(
                array=[[0.0, 1.0, 1.5]], pixel_scales=1.0
            )
            hyper_galaxy_image_1 = ag.Array2D.manual_native(
                array=[[1.0, 1.0, 1.5]], pixel_scales=1.0
            )

            galaxy_0 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_0,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image_0,
            )

            galaxy_1 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_1,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image_1,
            )

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0, galaxy_1])

            hyper_noise_maps = plane.hyper_noise_maps_of_galaxies_from_noise_map(
                noise_map=noise_map
            )

            assert (hyper_noise_maps[0].slim == np.array([0.0, 2.0, 3.0])).all()
            assert hyper_noise_maps[1].slim == pytest.approx(
                np.array([0.73468, (2.0 * 0.75) ** 2.0, 3.0 ** 2.0]), 1.0e-4
            )

        def test__hyper_noise_maps_are_same_as_hyper_galaxy_calculation(self):
            noise_map = ag.Array2D.manual_native(
                array=[[5.0, 3.0, 1.0]], pixel_scales=1.0
            )

            hyper_model_image = ag.Array2D.manual_native(
                array=[[2.0, 4.0, 10.0]], pixel_scales=1.0
            )
            hyper_galaxy_image = ag.Array2D.manual_native(
                array=[[1.0, 5.0, 8.0]], pixel_scales=1.0
            )

            hyper_galaxy_0 = ag.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = ag.HyperGalaxy(contribution_factor=10.0)

            contribution_map_0 = hyper_galaxy_0.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            contribution_map_1 = hyper_galaxy_1.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            hyper_noise_map_0 = hyper_galaxy_0.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map_0
            )

            hyper_noise_map_1 = hyper_galaxy_1.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map_1
            )

            galaxy_0 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_0,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            galaxy_1 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_1,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0])

            hyper_noise_maps = plane.hyper_noise_maps_of_galaxies_from_noise_map(
                noise_map=noise_map
            )
            assert (hyper_noise_maps[0].slim == hyper_noise_map_0).all()

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_1])

            hyper_noise_maps = plane.hyper_noise_maps_of_galaxies_from_noise_map(
                noise_map=noise_map
            )
            assert (hyper_noise_maps[0].slim == hyper_noise_map_1).all()

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_1, galaxy_0])

            hyper_noise_maps = plane.hyper_noise_maps_of_galaxies_from_noise_map(
                noise_map=noise_map
            )
            assert (hyper_noise_maps[0].slim == hyper_noise_map_1).all()
            assert (hyper_noise_maps[1].slim == hyper_noise_map_0).all()

        def test__hyper_noise_maps_are_none_for_galaxy_without_hyper_galaxy(self):
            noise_map = ag.Array2D.manual_native(
                array=[[5.0, 3.0, 1.0]], pixel_scales=1.0
            )

            hyper_model_image = ag.Array2D.manual_native(
                array=[[2.0, 4.0, 10.0]], pixel_scales=1.0
            )
            hyper_galaxy_image = ag.Array2D.manual_native(
                array=[[1.0, 5.0, 8.0]], pixel_scales=1.0
            )

            hyper_galaxy_0 = ag.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = ag.HyperGalaxy(contribution_factor=10.0)

            contribution_map_0 = hyper_galaxy_0.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            contribution_map_1 = hyper_galaxy_1.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            hyper_noise_map_0 = hyper_galaxy_0.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map_0
            )

            hyper_noise_map_1 = hyper_galaxy_1.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map_1
            )

            galaxy_0 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_0,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            galaxy_1 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_1,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0, ag.Galaxy(redshift=0.5)])

            hyper_noise_maps = plane.hyper_noise_maps_of_galaxies_from_noise_map(
                noise_map=noise_map
            )
            assert (hyper_noise_maps[0].slim == hyper_noise_map_0).all()
            assert hyper_noise_maps[1].slim == np.zeros(shape=(3, 1))

            plane = ag.Plane(redshift=0.5, galaxies=[ag.Galaxy(redshift=0.5), galaxy_1])

            hyper_noise_maps = plane.hyper_noise_maps_of_galaxies_from_noise_map(
                noise_map=noise_map
            )
            assert hyper_noise_maps[0].slim == np.zeros(shape=(3, 1))
            assert (hyper_noise_maps[1].slim == hyper_noise_map_1).all()

            plane = ag.Plane(
                redshift=0.5,
                galaxies=[
                    ag.Galaxy(redshift=0.5),
                    galaxy_1,
                    galaxy_0,
                    ag.Galaxy(redshift=0.5),
                ],
            )

            hyper_noise_maps = plane.hyper_noise_maps_of_galaxies_from_noise_map(
                noise_map=noise_map
            )
            assert hyper_noise_maps[0].slim == np.zeros(shape=(3, 1))
            assert (hyper_noise_maps[1].slim == hyper_noise_map_1).all()
            assert (hyper_noise_maps[2].slim == hyper_noise_map_0).all()
            assert hyper_noise_maps[3].slim == np.zeros(shape=(3, 1))

        def test__hyper_noise_map_from_noise_map__is_sum_of_galaxy_hyper_noise_maps__filters_nones(
            self,
        ):
            noise_map = ag.Array2D.manual_native(
                array=[[5.0, 3.0, 1.0]], pixel_scales=1.0
            )

            hyper_model_image = ag.Array2D.manual_native(
                array=[[2.0, 4.0, 10.0]], pixel_scales=1.0
            )
            hyper_galaxy_image = ag.Array2D.manual_native(
                array=[[1.0, 5.0, 8.0]], pixel_scales=1.0
            )

            hyper_galaxy_0 = ag.HyperGalaxy(contribution_factor=5.0)
            hyper_galaxy_1 = ag.HyperGalaxy(contribution_factor=10.0)

            contribution_map_0 = hyper_galaxy_0.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            contribution_map_1 = hyper_galaxy_1.contribution_map_from_hyper_images(
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            hyper_noise_map_0 = hyper_galaxy_0.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map_0
            )

            hyper_noise_map_1 = hyper_galaxy_1.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map_1
            )

            galaxy_0 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_0,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            galaxy_1 = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyper_galaxy_1,
                hyper_model_image=hyper_model_image,
                hyper_galaxy_image=hyper_galaxy_image,
            )

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0])

            hyper_noise_map = plane.hyper_noise_map_from_noise_map(noise_map=noise_map)
            assert (hyper_noise_map.slim == hyper_noise_map_0).all()

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_1])

            hyper_noise_map = plane.hyper_noise_map_from_noise_map(noise_map=noise_map)
            assert (hyper_noise_map.slim == hyper_noise_map_1).all()

            plane = ag.Plane(redshift=0.5, galaxies=[galaxy_1, galaxy_0])

            hyper_noise_map = plane.hyper_noise_map_from_noise_map(noise_map=noise_map)
            assert (hyper_noise_map.slim == hyper_noise_map_0 + hyper_noise_map_1).all()

            plane = ag.Plane(
                redshift=0.5,
                galaxies=[
                    ag.Galaxy(redshift=0.5),
                    galaxy_1,
                    galaxy_0,
                    ag.Galaxy(redshift=0.5),
                ],
            )

            hyper_noise_map = plane.hyper_noise_map_from_noise_map(noise_map=noise_map)
            assert (hyper_noise_map.slim == hyper_noise_map_0 + hyper_noise_map_1).all()

        def test__plane_has_no_hyper_galaxies__hyper_noise_map_function_returns_none(
            self,
        ):
            noise_map = ag.Array2D.manual_native(
                array=[[5.0, 3.0, 1.0]], pixel_scales=1.0
            )

            plane = ag.Plane(redshift=0.5, galaxies=[ag.Galaxy(redshift=0.5)])
            hyper_noise_map = plane.hyper_noise_map_from_noise_map(noise_map=noise_map)

            assert hyper_noise_map == np.zeros((3, 1))


class TestPlane:
    class TestTracedGrid:
        def test__traced_grid_same_as_manual_deflections_calc_via_galaxy___use_multiple_galaxies(
            self, sub_grid_7x7
        ):
            # Overwrite one value so intensity in each pixel is different
            sub_grid_7x7[5] = np.array([2.0, 2.0])

            g0 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=1.0),
            )
            g1 = ag.Galaxy(
                redshift=0.5,
                mass_profile=ag.mp.SphericalIsothermal(einstein_radius=2.0),
            )

            g0_deflections = g0.deflections_from_grid(grid=sub_grid_7x7)

            g1_deflections = g1.deflections_from_grid(grid=sub_grid_7x7)

            traced_grid = sub_grid_7x7 - (g0_deflections + g1_deflections)

            plane = ag.Plane(galaxies=[g0, g1], redshift=None)

            plane_traced_grid = plane.traced_grid_from_grid(grid=sub_grid_7x7)

            assert plane_traced_grid == pytest.approx(traced_grid, 1.0e-4)

        def test__traced_grid_numerics__uses_deflections__x2_sis_galaxies(
            self, sub_grid_7x7_simple, gal_x1_mp
        ):
            plane = ag.Plane(galaxies=[gal_x1_mp, gal_x1_mp], redshift=None)

            traced_grid = plane.traced_grid_from_grid(grid=sub_grid_7x7_simple)

            assert traced_grid[0] == pytest.approx(
                np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3
            )
            assert traced_grid[1] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
            assert traced_grid[2] == pytest.approx(
                np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3
            )
            assert traced_grid[3] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)

        def test__traced_grid__grid_is_positions__uses_deflections__x2_sis_galaxies(
            self, gal_x1_mp
        ):

            positions = ag.Grid2DIrregularGrouped(grid=[[(1.0, 1.0), (1.0, 0.0)]])

            plane = ag.Plane(galaxies=[gal_x1_mp, gal_x1_mp], redshift=None)

            traced_grid = plane.traced_grid_from_grid(grid=positions)

            assert traced_grid.in_grouped_list[0][0] == pytest.approx(
                (1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707), 1e-3
            )
            assert traced_grid.in_grouped_list[0][1] == pytest.approx((-1.0, 0.0), 1e-3)

        def test__plane_has_no_galaxies__traced_grid_is_input_grid_of_sub_grid_7x7(
            self, sub_grid_7x7
        ):
            plane = ag.Plane(galaxies=[], redshift=1.0)

            traced_grid = plane.traced_grid_from_grid(grid=sub_grid_7x7)

            assert (traced_grid == sub_grid_7x7).all()

    class TestGalaxies:
        def test__no_galaxies__raises_exception_if_no_plane_redshift_input(self):
            plane = ag.Plane(galaxies=[], redshift=0.5)
            assert plane.redshift == 0.5

            with pytest.raises(exc.PlaneException):
                ag.Plane(galaxies=[])

        def test__galaxy_redshifts_gives_list_of_redshifts(self):
            g0 = ag.Galaxy(redshift=1.0)
            g1 = ag.Galaxy(redshift=1.0)
            g2 = ag.Galaxy(redshift=1.0)

            plane = ag.Plane(galaxies=[g0, g1, g2])

            assert plane.redshift == 1.0
            assert plane.galaxy_redshifts == [1.0, 1.0, 1.0]


class TestPlaneImage:
    def test__compute_xticks_from_grid_correctly(self):

        array = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=(5.0, 1.0))

        plane_image = plane.PlaneImage(array=array, grid=None)
        assert plane_image.xticks == pytest.approx(
            np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3
        )

        array = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=(5.0, 0.5))

        plane_image = plane.PlaneImage(array=array, grid=None)
        assert plane_image.xticks == pytest.approx(
            np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
        )

        array = ag.Array2D.ones(shape_native=(1, 6), pixel_scales=(5.0, 1.0))

        plane_image = plane.PlaneImage(array=array, grid=None)
        assert plane_image.xticks == pytest.approx(
            np.array([-3.0, -1.0, 1.0, 3.0]), 1e-2
        )

    def test__compute_yticks_from_grid_correctly(self):

        array = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=(1.0, 5.0))

        plane_image = plane.PlaneImage(array=array, grid=None)
        assert plane_image.yticks == pytest.approx(
            np.array([-1.5, -0.5, 0.5, 1.5]), 1e-3
        )

        array = ag.Array2D.ones(shape_native=(3, 3), pixel_scales=(0.5, 5.0))

        plane_image = plane.PlaneImage(array=array, grid=None)
        assert plane_image.yticks == pytest.approx(
            np.array([-0.75, -0.25, 0.25, 0.75]), 1e-3
        )

        array = ag.Array2D.ones(shape_native=(6, 1), pixel_scales=(1.0, 5.0))

        plane_image = plane.PlaneImage(array=array, grid=None)
        assert plane_image.yticks == pytest.approx(
            np.array([-3.0, -1.0, 1.0, 3.0]), 1e-2
        )


class TestDecorators:
    def test__grid_iterate_in__iterates_grid_correctly(self, gal_x1_lp):

        mask = ag.Mask2D.manual(
            mask=[
                [True, True, True, True, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, False, False, False, True],
                [True, True, True, True, True],
            ],
            pixel_scales=(1.0, 1.0),
            origin=(0.001, 0.001),
        )

        grid = ag.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=1.0, sub_steps=[2]
        )

        plane = ag.Plane(galaxies=[gal_x1_lp], redshift=None)

        image = plane.image_from_grid(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = ag.Grid2D.from_mask(mask=mask_sub_2)
        image_sub_2 = plane.image_from_grid(grid=grid_sub_2).slim_binned

        assert (image == image_sub_2).all()

        grid = ag.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.95, sub_steps=[2, 4, 8]
        )

        galaxy = ag.Galaxy(
            redshift=0.5,
            light=ag.lp.EllipticalSersic(centre=(0.08, 0.08), intensity=1.0),
        )

        plane = ag.Plane(galaxies=[galaxy])

        image = plane.image_from_grid(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = ag.Grid2D.from_mask(mask=mask_sub_4)
        image_sub_4 = plane.image_from_grid(grid=grid_sub_4).slim_binned

        assert image[0] == image_sub_4[0]

        mask_sub_8 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)
        grid_sub_8 = ag.Grid2D.from_mask(mask=mask_sub_8)
        image_sub_8 = plane.image_from_grid(grid=grid_sub_8).slim_binned

        assert image[4] == image_sub_8[4]

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

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllipticalIsothermal(centre=(0.08, 0.08), einstein_radius=1.0),
        )

        plane = ag.Plane(galaxies=[galaxy], redshift=None)

        deflections = plane.deflections_from_grid(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = ag.Grid2D.from_mask(mask=mask_sub_2)
        deflections_sub_2 = galaxy.deflections_from_grid(grid=grid_sub_2).slim_binned

        assert (deflections == deflections_sub_2).all()

        grid = ag.Grid2DIterate.from_mask(
            mask=mask, fractional_accuracy=0.99, sub_steps=[2, 4, 8]
        )

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllipticalIsothermal(centre=(0.08, 0.08), einstein_radius=1.0),
        )

        plane = ag.Plane(galaxies=[galaxy], redshift=None)

        deflections = plane.deflections_from_grid(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = ag.Grid2D.from_mask(mask=mask_sub_4)
        deflections_sub_4 = galaxy.deflections_from_grid(grid=grid_sub_4).slim_binned

        assert deflections[0, 0] == deflections_sub_4[0, 0]

        mask_sub_8 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)
        grid_sub_8 = ag.Grid2D.from_mask(mask=mask_sub_8)
        deflections_sub_8 = galaxy.deflections_from_grid(grid=grid_sub_8).slim_binned

        assert deflections[4, 0] == deflections_sub_8[4, 0]

    def test__grid_interp_in__interps_based_on_intepolate_config(self):
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

        grid = ag.Grid2D.from_mask(mask=mask)

        grid_interp = ag.Grid2DInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

        light_profile = ag.lp.EllipticalSersic(intensity=1.0)
        light_profile_interp = ag.lp.SphericalSersic(intensity=1.0)

        image_no_interp = light_profile.image_from_grid(grid=grid)

        array_interp = light_profile.image_from_grid(grid=grid_interp.grid_interp)
        image_interp = grid_interp.interpolated_array_from_array_interp(
            array_interp=array_interp
        )

        galaxy = ag.Galaxy(
            redshift=0.5, light=light_profile_interp, light_0=light_profile
        )

        plane = ag.Plane(galaxies=[galaxy])

        image = plane.image_from_grid(grid=grid_interp)

        assert (image == image_no_interp + image_interp).all()

        mass_profile = ag.mp.EllipticalIsothermal(einstein_radius=1.0)
        mass_profile_interp = ag.mp.SphericalIsothermal(einstein_radius=1.0)

        convergence_no_interp = mass_profile.convergence_from_grid(grid=grid)

        array_interp = mass_profile_interp.convergence_from_grid(
            grid=grid_interp.grid_interp
        )
        convergence_interp = grid_interp.interpolated_array_from_array_interp(
            array_interp=array_interp
        )

        galaxy = ag.Galaxy(redshift=0.5, mass=mass_profile_interp, mass_0=mass_profile)

        plane = ag.Plane(galaxies=[galaxy])

        convergence = plane.convergence_from_grid(grid=grid_interp)

        assert (convergence == convergence_no_interp + convergence_interp).all()

        potential_no_interp = mass_profile.potential_from_grid(grid=grid)

        array_interp = mass_profile_interp.potential_from_grid(
            grid=grid_interp.grid_interp
        )
        potential_interp = grid_interp.interpolated_array_from_array_interp(
            array_interp=array_interp
        )

        galaxy = ag.Galaxy(redshift=0.5, mass=mass_profile_interp, mass_0=mass_profile)

        plane = ag.Plane(galaxies=[galaxy])

        potential = plane.potential_from_grid(grid=grid_interp)

        assert (potential == potential_no_interp + potential_interp).all()

        deflections_no_interp = mass_profile.deflections_from_grid(grid=grid)

        grid_interp_0 = mass_profile_interp.deflections_from_grid(
            grid=grid_interp.grid_interp
        )
        deflections_interp = grid_interp.interpolated_grid_from_grid_interp(
            grid_interp=grid_interp_0
        )

        galaxy = ag.Galaxy(redshift=0.5, mass=mass_profile_interp, mass_0=mass_profile)

        plane = ag.Plane(galaxies=[galaxy])

        deflections = plane.deflections_from_grid(grid=grid_interp)

        assert (deflections == deflections_no_interp + deflections_interp).all()


class TestRegression:
    def test__centre_of_profile_in_right_place(self):
        grid = ag.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=ag.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        plane = ag.Plane(galaxies=[galaxy])

        convergence = plane.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = plane.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = plane.deflections_from_grid(grid=grid)
        assert deflections.native[1, 4, 0] > 0
        assert deflections.native[2, 4, 0] < 0
        assert deflections.native[1, 4, 1] > 0
        assert deflections.native[1, 3, 1] < 0

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=ag.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        plane = ag.Plane(galaxies=[galaxy])

        convergence = plane.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = plane.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = plane.deflections_from_grid(grid=grid)
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

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=ag.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        plane = ag.Plane(galaxies=[galaxy])

        convergence = plane.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = plane.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = plane.deflections_from_grid(grid=grid)
        assert deflections.native[1, 4, 0] >= 0
        assert deflections.native[2, 4, 0] <= 0
        assert deflections.native[1, 4, 1] >= 0
        assert deflections.native[1, 3, 1] <= 0

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        plane = ag.Plane(galaxies=[galaxy])

        convergence = plane.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(
            convergence.native.argmax(), convergence.shape_native
        )
        assert max_indexes == (1, 4)

        potential = plane.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(
            potential.native.argmin(), potential.shape_native
        )
        assert max_indexes == (1, 4)

        deflections = plane.deflections_from_grid(grid=grid)
        assert deflections.native[1, 4, 0] >= 0
        assert deflections.native[2, 4, 0] <= 0
        assert deflections.native[1, 4, 1] >= 0
        assert deflections.native[1, 3, 1] <= 0
