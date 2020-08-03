import os

from autoconf import conf
import autogalaxy as ag
import numpy as np
import pytest
from autogalaxy import exc
from test_autogalaxy import mock


class TestUnits:
    def test__light_profiles_conversions(self):

        profile_0 = ag.lp.EllipticalGaussian(
            centre=(
                ag.dim.Length(value=3.0, unit_length="arcsec"),
                ag.dim.Length(value=3.0, unit_length="arcsec"),
            ),
            intensity=ag.dim.Luminosity(value=2.0, unit_luminosity="eps"),
        )

        profile_1 = ag.lp.EllipticalGaussian(
            centre=(
                ag.dim.Length(value=4.0, unit_length="arcsec"),
                ag.dim.Length(value=4.0, unit_length="arcsec"),
            ),
            intensity=ag.dim.Luminosity(value=5.0, unit_luminosity="eps"),
        )

        galaxy = ag.Galaxy(light_0=profile_0, light_1=profile_1, redshift=1.0)

        assert galaxy.light_0.centre == (3.0, 3.0)
        assert galaxy.light_0.unit_length == "arcsec"
        assert galaxy.light_0.intensity == 2.0
        assert galaxy.light_0.intensity.unit_luminosity == "eps"
        assert galaxy.light_1.centre == (4.0, 4.0)
        assert galaxy.light_1.unit_length == "arcsec"
        assert galaxy.light_1.intensity == 5.0
        assert galaxy.light_1.intensity.unit_luminosity == "eps"

        galaxy = galaxy.new_object_with_units_converted(
            unit_length="kpc",
            kpc_per_arcsec=2.0,
            unit_luminosity="counts",
            exposure_time=0.5,
        )

        assert galaxy.light_0.centre == (6.0, 6.0)
        assert galaxy.light_0.unit_length == "kpc"
        assert galaxy.light_0.intensity == 1.0
        assert galaxy.light_0.intensity.unit_luminosity == "counts"
        assert galaxy.light_1.centre == (8.0, 8.0)
        assert galaxy.light_1.unit_length == "kpc"
        assert galaxy.light_1.intensity == 2.5
        assert galaxy.light_1.intensity.unit_luminosity == "counts"

    def test__mass_profiles_conversions(self):

        profile_0 = ag.mp.EllipticalSersic(
            centre=(
                ag.dim.Length(value=3.0, unit_length="arcsec"),
                ag.dim.Length(value=3.0, unit_length="arcsec"),
            ),
            intensity=ag.dim.Luminosity(value=2.0, unit_luminosity="eps"),
            mass_to_light_ratio=ag.dim.MassOverLuminosity(
                value=5.0, unit_mass="angular", unit_luminosity="eps"
            ),
        )

        profile_1 = ag.mp.EllipticalSersic(
            centre=(
                ag.dim.Length(value=4.0, unit_length="arcsec"),
                ag.dim.Length(value=4.0, unit_length="arcsec"),
            ),
            intensity=ag.dim.Luminosity(value=5.0, unit_luminosity="eps"),
            mass_to_light_ratio=ag.dim.MassOverLuminosity(
                value=10.0, unit_mass="angular", unit_luminosity="eps"
            ),
        )

        galaxy = ag.Galaxy(mass_0=profile_0, mass_1=profile_1, redshift=1.0)

        assert galaxy.mass_0.centre == (3.0, 3.0)
        assert galaxy.mass_0.unit_length == "arcsec"
        assert galaxy.mass_0.intensity == 2.0
        assert galaxy.mass_0.intensity.unit_luminosity == "eps"
        assert galaxy.mass_0.mass_to_light_ratio == 5.0
        assert galaxy.mass_0.mass_to_light_ratio.unit_mass == "angular"
        assert galaxy.mass_1.centre == (4.0, 4.0)
        assert galaxy.mass_1.unit_length == "arcsec"
        assert galaxy.mass_1.intensity == 5.0
        assert galaxy.mass_1.intensity.unit_luminosity == "eps"
        assert galaxy.mass_1.mass_to_light_ratio == 10.0
        assert galaxy.mass_1.mass_to_light_ratio.unit_mass == "angular"

        galaxy = galaxy.new_object_with_units_converted(
            unit_length="kpc",
            kpc_per_arcsec=2.0,
            unit_luminosity="counts",
            exposure_time=0.5,
            unit_mass="solMass",
            critical_surface_density=3.0,
        )

        assert galaxy.mass_0.centre == (6.0, 6.0)
        assert galaxy.mass_0.unit_length == "kpc"
        assert galaxy.mass_0.intensity == 1.0
        assert galaxy.mass_0.intensity.unit_luminosity == "counts"
        assert galaxy.mass_0.mass_to_light_ratio == 30.0
        assert galaxy.mass_0.mass_to_light_ratio.unit_mass == "solMass"
        assert galaxy.mass_1.centre == (8.0, 8.0)
        assert galaxy.mass_1.unit_length == "kpc"
        assert galaxy.mass_1.intensity == 2.5
        assert galaxy.mass_1.intensity.unit_luminosity == "counts"
        assert galaxy.mass_1.mass_to_light_ratio == 60.0
        assert galaxy.mass_1.mass_to_light_ratio.unit_mass == "solMass"

    def test__galaxy_keeps_attributes(self):

        profile_0 = ag.lp.EllipticalGaussian(
            centre=(
                ag.dim.Length(value=3.0, unit_length="arcsec"),
                ag.dim.Length(value=3.0, unit_length="arcsec"),
            ),
            intensity=ag.dim.Luminosity(value=2.0, unit_luminosity="eps"),
        )

        galaxy = ag.Galaxy(
            light_0=profile_0, redshift=1.0, pixelization=1, regularization=2
        )

        assert galaxy.redshift == 1.0
        assert galaxy.pixelization == 1
        assert galaxy.regularization == 2

        galaxy = galaxy.new_object_with_units_converted(
            unit_length="kpc",
            kpc_per_arcsec=2.0,
            unit_luminosity="counts",
            exposure_time=0.5,
        )

        assert galaxy.redshift == 1.0
        assert galaxy.pixelization == 1
        assert galaxy.regularization == 2


class TestLightProfiles:
    class TestProfileImage:
        def test__no_light_profiles__image_returned_as_0s_of_shape_grid(
            self, sub_grid_7x7
        ):
            galaxy = ag.Galaxy(redshift=0.5)

            image = galaxy.image_from_grid(grid=sub_grid_7x7)

            assert (image == np.zeros(shape=sub_grid_7x7.sub_shape_1d)).all()

        def test__using_no_light_profiles__check_reshaping_decorator_of_returned_image(
            self, sub_grid_7x7
        ):
            galaxy = ag.Galaxy(redshift=0.5)

            image = galaxy.image_from_grid(grid=sub_grid_7x7)

            assert (image.in_2d_binned == np.zeros(shape=(7, 7))).all()

            image = galaxy.image_from_grid(grid=sub_grid_7x7)

            assert (image == np.zeros(shape=sub_grid_7x7.sub_shape_1d)).all()

            image = galaxy.image_from_grid(grid=sub_grid_7x7)

            assert (
                image.in_1d_binned == np.zeros(shape=sub_grid_7x7.sub_shape_1d // 4)
            ).all()

        def test__galaxies_with_x1_and_x2_light_profiles__image_is_same_individual_profiles(
            self, lp_0, gal_x1_lp, lp_1, gal_x2_lp
        ):
            lp_image = lp_0.image_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_lp_image = gal_x1_lp.image_from_grid(grid=np.array([[1.05, -0.55]]))

            assert lp_image == gal_lp_image

            lp_image = lp_0.image_from_grid(grid=np.array([[1.05, -0.55]]))
            lp_image += lp_1.image_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_image = gal_x2_lp.image_from_grid(grid=np.array([[1.05, -0.55]]))

            assert lp_image == gal_image

        def test__coordinates_in__coordinates_out(
            self, lp_0, gal_x1_lp, lp_1, gal_x2_lp
        ):

            lp_image = lp_0.image_from_grid(grid=ag.GridCoordinates([[(1.05, -0.55)]]))

            gal_lp_image = gal_x1_lp.image_from_grid(
                grid=ag.GridCoordinates([[(1.05, -0.55)]])
            )

            assert lp_image.in_list[0][0] == gal_lp_image.in_list[0][0]

        def test__sub_grid_in__grid_is_mapped_to_image_grid_by_wrapper_by_binning_sum_of_light_profile_values(
            self, sub_grid_7x7, gal_x2_lp
        ):

            lp_0_image = gal_x2_lp.light_profile_0.image_from_grid(grid=sub_grid_7x7)

            lp_1_image = gal_x2_lp.light_profile_1.image_from_grid(grid=sub_grid_7x7)

            lp_image = lp_0_image + lp_1_image

            lp_image_0 = (lp_image[0] + lp_image[1] + lp_image[2] + lp_image[3]) / 4.0

            lp_image_1 = (lp_image[4] + lp_image[5] + lp_image[6] + lp_image[7]) / 4.0

            gal_image = gal_x2_lp.image_from_grid(grid=sub_grid_7x7)

            assert gal_image.in_1d_binned[0] == lp_image_0
            assert gal_image.in_1d_binned[1] == lp_image_1

    class TestLuminosityWithin:
        def test__two_profile_galaxy__is_sum_of_individual_profiles(
            self, lp_0, lp_1, gal_x1_lp, gal_x2_lp
        ):

            radius = ag.dim.Length(0.5, "arcsec")

            lp_luminosity = lp_0.luminosity_within_circle_in_units(
                radius=radius, unit_luminosity="eps"
            )
            gal_luminosity = gal_x1_lp.luminosity_within_circle_in_units(
                radius=radius, unit_luminosity="eps"
            )

            assert lp_luminosity == gal_luminosity

        def test__radius_unit_conversions__multiply_by_kpc_per_arcsec(
            self, lp_0, gal_x1_lp
        ):
            cosmology = mock.MockCosmology(arcsec_per_kpc=0.5, kpc_per_arcsec=2.0)

            radius = ag.dim.Length(0.5, "arcsec")

            lp_luminosity_arcsec = lp_0.luminosity_within_circle_in_units(radius=radius)
            gal_luminosity_arcsec = gal_x1_lp.luminosity_within_circle_in_units(
                radius=radius
            )

            assert lp_luminosity_arcsec == gal_luminosity_arcsec

            radius = ag.dim.Length(0.5, "kpc")

            lp_luminosity_kpc = lp_0.luminosity_within_circle_in_units(
                radius=radius, redshift_object=0.5, cosmology=cosmology
            )
            gal_luminosity_kpc = gal_x1_lp.luminosity_within_circle_in_units(
                radius=radius, cosmology=cosmology
            )

            assert lp_luminosity_kpc == gal_luminosity_kpc

        def test__luminosity_unit_conversions__multiply_by_exposure_time(
            self, lp_0, gal_x1_lp
        ):

            radius = 3.0

            lp_luminosity_counts = lp_0.luminosity_within_circle_in_units(
                radius=radius, unit_luminosity="counts", exposure_time=2.0
            )

            gal_luminosity_counts = gal_x1_lp.luminosity_within_circle_in_units(
                radius=radius, unit_luminosity="counts", exposure_time=2.0
            )

            assert lp_luminosity_counts == gal_luminosity_counts

        def test__no_light_profile__returns_none(self):
            gal_no_lp = ag.Galaxy(redshift=0.5, mass=ag.mp.SphericalIsothermal())

            assert gal_no_lp.luminosity_within_circle_in_units(radius=1.0) == None

    class TestSymmetricProfiles:
        def test_1d_symmetry(self):
            lp_0 = ag.lp.EllipticalSersic(centre=(0.0, 0.0), intensity=1.0)

            lp_1 = ag.lp.EllipticalSersic(centre=(100.0, 0.0), intensity=1.0)

            gal_x2_lp = ag.Galaxy(
                redshift=0.5, light_profile_0=lp_0, light_profile_1=lp_1
            )

            assert gal_x2_lp.image_from_grid(
                grid=np.array([[0.0, 0.0]])
            ) == pytest.approx(
                gal_x2_lp.image_from_grid(grid=np.array([[100.0, 0.0]])), 1.0e-4
            )

            assert gal_x2_lp.image_from_grid(
                grid=np.array([[49.0, 0.0]])
            ) == pytest.approx(
                gal_x2_lp.image_from_grid(grid=np.array([[51.0, 0.0]])), 1.0e-4
            )

        def test_2d_symmetry(self):
            lp_0 = ag.lp.EllipticalSersic(
                elliptical_comps=(0.0, 0.0),
                intensity=1.0,
                effective_radius=0.6,
                sersic_index=4.0,
            )

            lp_1 = ag.lp.EllipticalSersic(
                elliptical_comps=(0.0, 0.0),
                intensity=1.0,
                effective_radius=0.6,
                sersic_index=4.0,
                centre=(100, 0),
            )

            lp_2 = ag.lp.EllipticalSersic(
                elliptical_comps=(0.0, 0.0),
                intensity=1.0,
                effective_radius=0.6,
                sersic_index=4.0,
                centre=(0, 100),
            )

            lp_3 = ag.lp.EllipticalSersic(
                elliptical_comps=(0.0, 0.0),
                intensity=1.0,
                effective_radius=0.6,
                sersic_index=4.0,
                centre=(100, 100),
            )

            gal_x4_lp = ag.Galaxy(
                redshift=0.5,
                light_profile_0=lp_0,
                light_profile_1=lp_1,
                light_profile_3=lp_2,
                light_profile_4=lp_3,
            )

            assert gal_x4_lp.image_from_grid(
                grid=np.array([[49.0, 0.0]])
            ) == pytest.approx(
                gal_x4_lp.image_from_grid(grid=np.array([[51.0, 0.0]])), 1e-5
            )

            assert gal_x4_lp.image_from_grid(
                grid=np.array([[0.0, 49.0]])
            ) == pytest.approx(
                gal_x4_lp.image_from_grid(grid=np.array([[0.0, 51.0]])), 1e-5
            )

            assert gal_x4_lp.image_from_grid(
                grid=np.array([[100.0, 49.0]])
            ) == pytest.approx(
                gal_x4_lp.image_from_grid(grid=np.array([[100.0, 51.0]])), 1e-5
            )

            assert gal_x4_lp.image_from_grid(
                grid=np.array([[49.0, 49.0]])
            ) == pytest.approx(
                gal_x4_lp.image_from_grid(grid=np.array([[51.0, 51.0]])), 1e-5
            )

    class TestBlurredProfileImages:
        def test__blurred_image_from_grid_and_psf(
            self, sub_grid_7x7, blurring_grid_7x7, psf_3x3, convolver_7x7
        ):

            light_profile_0 = ag.lp.EllipticalSersic(intensity=2.0)
            light_profile_1 = ag.lp.EllipticalSersic(intensity=3.0)

            galaxy = ag.Galaxy(
                light_profile_0=light_profile_0,
                light_profile_1=light_profile_1,
                redshift=0.5,
            )

            image = galaxy.image_from_grid(grid=sub_grid_7x7)

            blurring_image = galaxy.image_from_grid(grid=blurring_grid_7x7)

            blurred_image = convolver_7x7.convolved_image_from_image_and_blurring_image(
                image=image.in_1d_binned, blurring_image=blurring_image.in_1d_binned
            )

            light_profile_blurred_image = galaxy.blurred_image_from_grid_and_psf(
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
            light_profile_0 = ag.lp.EllipticalSersic(intensity=2.0)
            light_profile_1 = ag.lp.EllipticalSersic(intensity=3.0)

            galaxy = ag.Galaxy(
                light_profile_0=light_profile_0,
                light_profile_1=light_profile_1,
                redshift=0.5,
            )

            image = galaxy.image_from_grid(grid=sub_grid_7x7)

            blurring_image = galaxy.image_from_grid(grid=blurring_grid_7x7)

            blurred_image = convolver_7x7.convolved_image_from_image_and_blurring_image(
                image=image.in_1d_binned, blurring_image=blurring_image.in_1d_binned
            )

            light_profile_blurred_image = galaxy.blurred_image_from_grid_and_convolver(
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

    class TestVisibilities:
        def test__visibilities_from_grid_and_transformer(
            self, sub_grid_7x7, transformer_7x7_7
        ):

            light_profile_0 = ag.lp.EllipticalSersic(intensity=2.0)
            light_profile_1 = ag.lp.EllipticalSersic(intensity=3.0)

            image = (
                light_profile_0.image_from_grid(grid=sub_grid_7x7).in_1d_binned
                + light_profile_1.image_from_grid(grid=sub_grid_7x7).in_1d_binned
            )

            visibilities = transformer_7x7_7.visibilities_from_image(image=image)

            galaxy = ag.Galaxy(
                light_profile_0=light_profile_0,
                light_profile_1=light_profile_1,
                redshift=0.5,
            )

            galaxy_visibilities = galaxy.profile_visibilities_from_grid_and_transformer(
                grid=sub_grid_7x7, transformer=transformer_7x7_7
            )

            assert visibilities == pytest.approx(galaxy_visibilities, 1.0e-4)

    class TestLightProfileGeometry:
        def test__extracts_centres_correctly(self):

            galaxy = ag.Galaxy(redshift=0.5)

            assert galaxy.light_profile_centres == []

            galaxy = ag.Galaxy(
                redshift=0.5, lp_0=ag.lp.EllipticalLightProfile(centre=(0.0, 1.0))
            )

            assert galaxy.light_profile_centres.in_list == [[(0.0, 1.0)]]
            assert galaxy.light_profile_centres.as_dict["lp_0"] == [(0.0, 1.0)]

            galaxy = ag.Galaxy(
                redshift=0.5,
                lp_0=ag.lp.EllipticalLightProfile(centre=(0.0, 1.0)),
                lp_1=ag.lp.EllipticalLightProfile(centre=(2.0, 3.0)),
                lp_2=ag.lp.EllipticalLightProfile(centre=(4.0, 5.0)),
            )

            assert galaxy.light_profile_centres.in_list == [
                [(0.0, 1.0)],
                [(2.0, 3.0)],
                [(4.0, 5.0)],
            ]
            assert galaxy.light_profile_centres.as_dict["lp_0"] == [(0.0, 1.0)]
            assert galaxy.light_profile_centres.as_dict["lp_1"] == [(2.0, 3.0)]
            assert galaxy.light_profile_centres.as_dict["lp_2"] == [(4.0, 5.0)]

            galaxy = ag.Galaxy(
                redshift=0.5,
                lp_0=ag.lp.EllipticalLightProfile(centre=(0.0, 1.0)),
                lp_1=ag.lp.EllipticalLightProfile(centre=(2.0, 3.0)),
                mp_0=ag.mp.EllipticalMassProfile(centre=(-1.0, -2.0)),
                lp_2=ag.lp.EllipticalLightProfile(centre=(4.0, 5.0)),
            )

            assert galaxy.light_profile_centres.in_list == [
                [(0.0, 1.0)],
                [(2.0, 3.0)],
                [(4.0, 5.0)],
            ]
            assert galaxy.light_profile_centres.as_dict["lp_0"] == [(0.0, 1.0)]
            assert galaxy.light_profile_centres.as_dict["lp_1"] == [(2.0, 3.0)]
            assert galaxy.light_profile_centres.as_dict["lp_2"] == [(4.0, 5.0)]


class TestMassProfiles:
    class TestConvergence:
        def test__no_mass_profiles__convergence_returned_as_0s_of_shape_grid(
            self, sub_grid_7x7
        ):
            galaxy = ag.Galaxy(redshift=0.5)

            convergence = galaxy.convergence_from_grid(grid=sub_grid_7x7)

            assert (
                convergence.in_1d == np.zeros(shape=sub_grid_7x7.sub_shape_1d)
            ).all()

        def test__using_no_mass_profiles__check_reshaping_decorator_of_returned_convergence(
            self, sub_grid_7x7
        ):
            galaxy = ag.Galaxy(redshift=0.5)

            convergence = galaxy.convergence_from_grid(grid=sub_grid_7x7)

            assert (convergence.in_2d_binned == np.zeros(shape=(7, 7))).all()

            convergence = galaxy.convergence_from_grid(grid=sub_grid_7x7)

            assert (
                convergence.in_1d == np.zeros(shape=sub_grid_7x7.sub_shape_1d)
            ).all()

            convergence = galaxy.convergence_from_grid(grid=sub_grid_7x7)

            assert (
                convergence.in_1d_binned
                == np.zeros(shape=sub_grid_7x7.sub_shape_1d // 4)
            ).all()

        def test__galaxies_with_x1_and_x2_mass_profiles__convergence_is_same_individual_profiles(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):
            mp_convergence = mp_0.convergence_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_mp_convergence = gal_x1_mp.convergence_from_grid(
                grid=np.array([[1.05, -0.55]])
            )

            assert mp_convergence == gal_mp_convergence

            mp_convergence = mp_0.convergence_from_grid(grid=np.array([[1.05, -0.55]]))
            mp_convergence += mp_1.convergence_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_convergence = gal_x2_mp.convergence_from_grid(
                grid=np.array([[1.05, -0.55]])
            )

            assert mp_convergence == gal_convergence

        def test__coordinates_in__coordinates_out(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):

            mp_convergence = mp_0.convergence_from_grid(
                grid=ag.GridCoordinates([[(1.05, -0.55)]])
            )

            gal_mp_convergence = gal_x1_mp.convergence_from_grid(
                grid=ag.GridCoordinates([[(1.05, -0.55)]])
            )

            assert mp_convergence == gal_mp_convergence

            assert mp_convergence.in_list[0][0] == gal_mp_convergence.in_list[0][0]

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
            galaxy = ag.Galaxy(redshift=0.5)

            potential = galaxy.potential_from_grid(grid=sub_grid_7x7)

            assert (potential.in_1d == np.zeros(shape=sub_grid_7x7.sub_shape_1d)).all()

        def test__using_no_mass_profiles__check_reshaping_decorator_of_returned_potential(
            self, sub_grid_7x7
        ):
            galaxy = ag.Galaxy(redshift=0.5)

            potential = galaxy.potential_from_grid(grid=sub_grid_7x7)

            assert (potential.in_2d_binned == np.zeros(shape=(7, 7))).all()

            potential = galaxy.potential_from_grid(grid=sub_grid_7x7)

            assert (potential.in_1d == np.zeros(shape=sub_grid_7x7.sub_shape_1d)).all()

            potential = galaxy.potential_from_grid(grid=sub_grid_7x7)

            assert (
                potential.in_1d_binned == np.zeros(shape=sub_grid_7x7.sub_shape_1d // 4)
            ).all()

        def test__galaxies_with_x1_and_x2_mass_profiles__potential_is_same_individual_profiles(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):
            mp_potential = mp_0.potential_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_mp_potential = gal_x1_mp.potential_from_grid(
                grid=np.array([[1.05, -0.55]])
            )

            assert mp_potential == gal_mp_potential

            mp_potential = mp_0.potential_from_grid(grid=np.array([[1.05, -0.55]]))
            mp_potential += mp_1.potential_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_potential = gal_x2_mp.potential_from_grid(
                grid=np.array([[1.05, -0.55]])
            )

            assert mp_potential == gal_potential

        def test__coordinates_in__coordinates_out(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):

            mp_potential = mp_0.potential_from_grid(
                grid=ag.GridCoordinates([[(1.05, -0.55)]])
            )

            gal_mp_potential = gal_x1_mp.potential_from_grid(
                grid=ag.GridCoordinates([[(1.05, -0.55)]])
            )

            assert mp_potential == gal_mp_potential

            assert mp_potential.in_list[0][0] == gal_mp_potential.in_list[0][0]

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
            galaxy = ag.Galaxy(redshift=0.5)

            deflections = galaxy.deflections_from_grid(grid=sub_grid_7x7)

            assert (
                deflections.in_1d == np.zeros(shape=(sub_grid_7x7.sub_shape_1d, 2))
            ).all()

        def test__using_no_mass_profiles__check_reshaping_decorator_of_returned_deflections(
            self, sub_grid_7x7
        ):
            galaxy = ag.Galaxy(redshift=0.5)

            deflections = galaxy.deflections_from_grid(grid=sub_grid_7x7)

            assert (deflections.in_2d_binned == np.zeros(shape=(7, 7, 2))).all()

            deflections = galaxy.deflections_from_grid(grid=sub_grid_7x7)

            assert (
                deflections.in_1d == np.zeros(shape=(sub_grid_7x7.sub_shape_1d, 2))
            ).all()

            deflections = galaxy.deflections_from_grid(grid=sub_grid_7x7)

            assert (
                deflections.in_1d_binned
                == np.zeros(shape=(sub_grid_7x7.sub_shape_1d // 4, 2))
            ).all()

        def test__galaxies_with_x1_and_x2_mass_profiles__deflections_is_same_individual_profiles(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):
            mp_deflections = mp_0.deflections_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_mp_deflections = gal_x1_mp.deflections_from_grid(
                grid=np.array([[1.05, -0.55]])
            )

            assert (mp_deflections == gal_mp_deflections).all()

            mp_deflections = mp_0.deflections_from_grid(grid=np.array([[1.05, -0.55]]))
            mp_deflections += mp_1.deflections_from_grid(grid=np.array([[1.05, -0.55]]))

            gal_deflections = gal_x2_mp.deflections_from_grid(
                grid=np.array([[1.05, -0.55]])
            )

            assert (mp_deflections == gal_deflections).all()

        def test__coordinates_in__coordinates_out(
            self, mp_0, gal_x1_mp, mp_1, gal_x2_mp
        ):

            mp_deflections = mp_0.deflections_from_grid(
                grid=ag.GridCoordinates([[(1.05, -0.55)]])
            )

            gal_mp_deflections = gal_x1_mp.deflections_from_grid(
                grid=ag.GridCoordinates([[(1.05, -0.55)]])
            )

            assert type(gal_mp_deflections) == ag.GridCoordinates
            assert (
                mp_deflections.in_list[0][0][0] == gal_mp_deflections.in_list[0][0][0]
            )
            assert (
                mp_deflections.in_list[0][0][1] == gal_mp_deflections.in_list[0][0][1]
            )

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
            radius = ag.dim.Length(0.5, "arcsec")

            mp_mass = mp_0.mass_within_circle_in_units(
                radius=radius, unit_mass="angular"
            )

            gal_mass = gal_x1_mp.mass_within_circle_in_units(
                radius=radius, unit_mass="angular"
            )

            assert mp_mass == gal_mass

        def test__radius_unit_conversions__multiply_by_kpc_per_arcsec(
            self, mp_0, gal_x1_mp
        ):
            cosmology = mock.MockCosmology(
                arcsec_per_kpc=0.5, kpc_per_arcsec=2.0, critical_surface_density=1.0
            )

            radius = ag.dim.Length(0.5, "arcsec")

            mp_mass_arcsec = mp_0.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_object=0.5,
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

            radius = ag.dim.Length(0.5, "kpc")

            mp_mass_kpc = mp_0.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_object=0.5,
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
            cosmology = mock.MockCosmology(
                arcsec_per_kpc=1.0, kpc_per_arcsec=1.0, critical_surface_density=2.0
            )

            radius = ag.dim.Length(0.5, "arcsec")

            mp_mass_sol = mp_0.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_object=0.5,
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
            gal_no_mp = ag.Galaxy(redshift=0.5, light=ag.lp.SphericalSersic())

            with pytest.raises(exc.GalaxyException):
                gal_no_mp.mass_within_circle_in_units(radius=1.0)

    class TestStellar:
        def test__stellar_profiles__is_list_of_stellar_profiles(self):
            galaxy = ag.Galaxy(redshift=0.5)

            assert galaxy.stellar_profiles == []

            galaxy = ag.Galaxy(
                redshift=0.5,
                light=ag.lp.EllipticalGaussian(),
                mass=ag.mp.EllipticalIsothermal(),
            )

            assert galaxy.stellar_profiles == []

            stellar_0 = ag.lmp.EllipticalSersic()
            galaxy = ag.Galaxy(redshift=0.5, stellar_0=stellar_0)

            assert galaxy.stellar_profiles == [stellar_0]

            stellar_0 = ag.lmp.SphericalDevVaucouleurs()
            stellar_1 = ag.lmp.EllipticalDevVaucouleurs()
            stellar_2 = ag.lmp.EllipticalGaussian()
            galaxy = ag.Galaxy(
                redshift=0.5,
                stellar_0=stellar_0,
                stellar_1=stellar_1,
                stellar_2=stellar_2,
            )

            assert galaxy.stellar_profiles == [stellar_0, stellar_1, stellar_2]

        def test__stellar_mass_within_galaxy__is_sum_of_individual_profiles(
            self, smp_0, smp_1
        ):

            galaxy = ag.Galaxy(
                redshift=0.5,
                stellar_0=smp_0,
                non_stellar_profile=ag.mp.EllipticalIsothermal(einstein_radius=1.0),
            )

            radius = ag.dim.Length(0.5, "arcsec")

            stellar_mass_0 = smp_0.mass_within_circle_in_units(
                radius=radius, unit_mass="angular"
            )

            gal_mass = galaxy.stellar_mass_within_circle_in_units(
                radius=radius, unit_mass="angular"
            )

            assert stellar_mass_0 == gal_mass

            stellar_mass_0 = smp_0.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_object=0.5,
                redshift_source=1.0,
            )

            gal_mass = galaxy.stellar_mass_within_circle_in_units(
                radius=radius, unit_mass="solMass", redshift_source=1.0
            )

            assert stellar_mass_0 == gal_mass

            galaxy = ag.Galaxy(
                redshift=0.5,
                stellar_0=smp_0,
                stellar_1=smp_1,
                non_stellar_profile=ag.mp.EllipticalIsothermal(einstein_radius=1.0),
            )

            stellar_mass_1 = smp_1.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_object=0.5,
                redshift_source=1.0,
            )

            gal_mass = galaxy.stellar_mass_within_circle_in_units(
                radius=radius, unit_mass="solMass", redshift_source=1.0
            )

            assert stellar_mass_0 + stellar_mass_1 == gal_mass

            galaxy = ag.Galaxy(redshift=0.5)

            with pytest.raises(exc.GalaxyException):
                galaxy.stellar_mass_within_circle_in_units(radius=1.0)

        def test__stellar_fraction_at_radius(self, dmp_0, dmp_1, smp_0, smp_1):

            galaxy = ag.Galaxy(redshift=0.5, stellar_0=smp_0, dark_0=dmp_0)

            stellar_mass_0 = smp_0.mass_within_circle_in_units(radius=1.0)
            dark_mass_0 = dmp_0.mass_within_circle_in_units(radius=1.0)

            stellar_fraction = galaxy.stellar_fraction_at_radius(radius=1.0)

            assert stellar_fraction == pytest.approx(
                stellar_mass_0 / (dark_mass_0 + stellar_mass_0), 1.0e-4
            )

            galaxy = ag.Galaxy(
                redshift=0.5, stellar_0=smp_0, stellar_1=smp_1, dark_0=dmp_0
            )

            stellar_fraction = galaxy.stellar_fraction_at_radius(radius=1.0)
            stellar_mass_1 = smp_1.mass_within_circle_in_units(radius=1.0)

            assert stellar_fraction == pytest.approx(
                (stellar_mass_0 + stellar_mass_1)
                / (dark_mass_0 + stellar_mass_0 + stellar_mass_1),
                1.0e-4,
            )

            galaxy = ag.Galaxy(
                redshift=0.5,
                stellar_0=smp_0,
                stellar_1=smp_1,
                dark_0=dmp_0,
                dark_mass_1=dmp_1,
            )

            stellar_fraction = galaxy.stellar_fraction_at_radius(radius=1.0)
            dark_mass_1 = dmp_1.mass_within_circle_in_units(radius=1.0)

            assert stellar_fraction == pytest.approx(
                (stellar_mass_0 + stellar_mass_1)
                / (dark_mass_0 + dark_mass_1 + stellar_mass_0 + stellar_mass_1),
                1.0e-4,
            )

    class TestDark:
        def test__dark_profiles__is_list_of_dark_profiles(self):
            galaxy = ag.Galaxy(redshift=0.5)

            assert galaxy.dark_profiles == []

            galaxy = ag.Galaxy(
                redshift=0.5,
                light=ag.lp.EllipticalGaussian(),
                mass=ag.mp.EllipticalIsothermal(),
            )

            assert galaxy.dark_profiles == []

            dark_0 = ag.mp.SphericalNFW()
            galaxy = ag.Galaxy(redshift=0.5, dark_0=dark_0)

            assert galaxy.dark_profiles == [dark_0]

            dark_0 = ag.mp.SphericalNFW()
            dark_1 = ag.mp.EllipticalNFW()
            dark_2 = ag.mp.EllipticalGeneralizedNFW()
            galaxy = ag.Galaxy(
                redshift=0.5, dark_0=dark_0, dark_1=dark_1, dark_2=dark_2
            )

            assert galaxy.dark_profiles == [dark_0, dark_1, dark_2]

        def test__dark_mass_within_galaxy__is_sum_of_individual_profiles(
            self, dmp_0, dmp_1
        ):

            galaxy = ag.Galaxy(
                redshift=0.5,
                dark_0=dmp_0,
                non_dark_profile=ag.mp.EllipticalIsothermal(einstein_radius=1.0),
            )

            radius = ag.dim.Length(0.5, "arcsec")

            dark_mass_0 = dmp_0.mass_within_circle_in_units(
                radius=radius, unit_mass="angular"
            )

            gal_mass = galaxy.dark_mass_within_circle_in_units(
                radius=radius, unit_mass="angular"
            )

            assert dark_mass_0 == gal_mass

            dark_mass_0 = dmp_0.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_object=0.5,
                redshift_source=1.0,
            )

            gal_mass = galaxy.dark_mass_within_circle_in_units(
                radius=radius, unit_mass="solMass", redshift_source=1.0
            )

            assert dark_mass_0 == gal_mass

            galaxy = ag.Galaxy(
                redshift=0.5,
                dark_0=dmp_0,
                dark_1=dmp_1,
                non_dark_profile=ag.mp.EllipticalIsothermal(einstein_radius=1.0),
            )

            dark_mass_1 = dmp_1.mass_within_circle_in_units(
                radius=radius,
                unit_mass="solMass",
                redshift_object=0.5,
                redshift_source=1.0,
            )

            gal_mass = galaxy.dark_mass_within_circle_in_units(
                radius=radius, unit_mass="solMass", redshift_source=1.0
            )

            assert dark_mass_0 + dark_mass_1 == gal_mass

            galaxy = ag.Galaxy(redshift=0.5)

            with pytest.raises(exc.GalaxyException):
                galaxy.dark_mass_within_circle_in_units(radius=1.0)

        def test__dark_fraction_at_radius(self, dmp_0, dmp_1, smp_0, smp_1):

            galaxy = ag.Galaxy(redshift=0.5, dark_0=dmp_0, stellar_0=smp_0)

            stellar_mass_0 = smp_0.mass_within_circle_in_units(radius=1.0)
            dark_mass_0 = dmp_0.mass_within_circle_in_units(radius=1.0)

            dark_fraction = galaxy.dark_fraction_at_radius(radius=1.0)

            assert dark_fraction == dark_mass_0 / (stellar_mass_0 + dark_mass_0)

            galaxy = ag.Galaxy(
                redshift=0.5, dark_0=dmp_0, dark_1=dmp_1, stellar_0=smp_0
            )

            dark_fraction = galaxy.dark_fraction_at_radius(radius=1.0)
            dark_mass_1 = dmp_1.mass_within_circle_in_units(radius=1.0)

            assert dark_fraction == pytest.approx(
                (dark_mass_0 + dark_mass_1)
                / (stellar_mass_0 + dark_mass_0 + dark_mass_1),
                1.0e-4,
            )

            galaxy = ag.Galaxy(
                redshift=0.5,
                dark_0=dmp_0,
                dark_1=dmp_1,
                stellar_0=smp_0,
                stellar_mass_1=smp_1,
            )

            dark_fraction = galaxy.dark_fraction_at_radius(radius=1.0)
            stellar_mass_1 = smp_1.mass_within_circle_in_units(radius=1.0)

            assert dark_fraction == pytest.approx(
                (dark_mass_0 + dark_mass_1)
                / (stellar_mass_0 + stellar_mass_1 + dark_mass_0 + dark_mass_1),
                1.0e-4,
            )

    class TestSymmetricProfiles:
        def test_1d_symmetry(self):
            mp_0 = ag.mp.EllipticalIsothermal(
                elliptical_comps=(0.333333, 0.0), einstein_radius=1.0
            )

            mp_1 = ag.mp.EllipticalIsothermal(
                centre=(100, 0), elliptical_comps=(0.333333, 0.0), einstein_radius=1.0
            )

            gal_x4_mp = ag.Galaxy(
                redshift=0.5, mass_profile_0=mp_0, mass_profile_1=mp_1
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=np.array([[1.0, 0.0]])
            ) == pytest.approx(
                gal_x4_mp.convergence_from_grid(grid=np.array([[99.0, 0.0]])), 1.0e-4
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=np.array([[49.0, 0.0]])
            ) == pytest.approx(
                gal_x4_mp.convergence_from_grid(grid=np.array([[51.0, 0.0]])), 1.0e-4
            )

            assert gal_x4_mp.potential_from_grid(
                grid=np.array([[1.0, 0.0]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=np.array([[99.0, 0.0]])), 1e-6
            )

            assert gal_x4_mp.potential_from_grid(
                grid=np.array([[49.0, 0.0]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=np.array([[51.0, 0.0]])), 1e-6
            )

            assert gal_x4_mp.deflections_from_grid(
                grid=np.array([[1.0, 0.0]])
            ) == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=np.array([[99.0, 0.0]])), 1e-6
            )

            assert gal_x4_mp.deflections_from_grid(
                grid=np.array([[49.0, 0.0]])
            ) == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=np.array([[51.0, 0.0]])), 1e-6
            )

        def test_2d_symmetry(self):
            mp_0 = ag.mp.SphericalIsothermal(einstein_radius=1.0)

            mp_1 = ag.mp.SphericalIsothermal(centre=(100, 0), einstein_radius=1.0)

            mp_2 = ag.mp.SphericalIsothermal(centre=(0, 100), einstein_radius=1.0)

            mp_3 = ag.mp.SphericalIsothermal(centre=(100, 100), einstein_radius=1.0)

            gal_x4_mp = ag.Galaxy(
                redshift=0.5,
                mass_profile_0=mp_0,
                mass_profile_1=mp_1,
                mass_profile_2=mp_2,
                mass_profile_3=mp_3,
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=np.array([[49.0, 0.0]])
            ) == pytest.approx(
                gal_x4_mp.convergence_from_grid(grid=np.array([[51.0, 0.0]])), 1e-5
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=np.array([[0.0, 49.0]])
            ) == pytest.approx(
                gal_x4_mp.convergence_from_grid(grid=np.array([[0.0, 51.0]])), 1e-5
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=np.array([[100.0, 49.0]])
            ) == pytest.approx(
                gal_x4_mp.convergence_from_grid(grid=np.array([[100.0, 51.0]])), 1e-5
            )

            assert gal_x4_mp.convergence_from_grid(
                grid=np.array([[49.0, 49.0]])
            ) == pytest.approx(
                gal_x4_mp.convergence_from_grid(grid=np.array([[51.0, 51.0]])), 1e-5
            )

            assert gal_x4_mp.potential_from_grid(
                grid=np.array([[49.0, 0.0]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=np.array([[51.0, 0.0]])), 1e-5
            )

            assert gal_x4_mp.potential_from_grid(
                grid=np.array([[0.0, 49.0]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=np.array([[0.0, 51.0]])), 1e-5
            )

            assert gal_x4_mp.potential_from_grid(
                grid=np.array([[100.0, 49.0]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=np.array([[100.0, 51.0]])), 1e-5
            )

            assert gal_x4_mp.potential_from_grid(
                grid=np.array([[49.0, 49.0]])
            ) == pytest.approx(
                gal_x4_mp.potential_from_grid(grid=np.array([[51.0, 51.0]])), 1e-5
            )

            assert -1.0 * gal_x4_mp.deflections_from_grid(grid=np.array([[49.0, 0.0]]))[
                0, 0
            ] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=np.array([[51.0, 0.0]]))[0, 0],
                1e-5,
            )

            assert 1.0 * gal_x4_mp.deflections_from_grid(grid=np.array([[0.0, 49.0]]))[
                0, 0
            ] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=np.array([[0.0, 51.0]]))[0, 0],
                1e-5,
            )

            assert 1.0 * gal_x4_mp.deflections_from_grid(
                grid=np.array([[100.0, 49.0]])
            )[0, 0] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=np.array([[100.0, 51.0]]))[0, 0],
                1e-5,
            )

            assert -1.0 * gal_x4_mp.deflections_from_grid(
                grid=np.array([[49.0, 49.0]])
            )[0, 0] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=np.array([[51.0, 51.0]]))[0, 0],
                1e-5,
            )

            assert 1.0 * gal_x4_mp.deflections_from_grid(grid=np.array([[49.0, 0.0]]))[
                0, 1
            ] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=np.array([[51.0, 0.0]]))[0, 1],
                1e-5,
            )

            assert -1.0 * gal_x4_mp.deflections_from_grid(grid=np.array([[0.0, 49.0]]))[
                0, 1
            ] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=np.array([[0.0, 51.0]]))[0, 1],
                1e-5,
            )

            assert -1.0 * gal_x4_mp.deflections_from_grid(
                grid=np.array([[100.0, 49.0]])
            )[0, 1] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=np.array([[100.0, 51.0]]))[0, 1],
                1e-5,
            )

            assert -1.0 * gal_x4_mp.deflections_from_grid(
                grid=np.array([[49.0, 49.0]])
            )[0, 1] == pytest.approx(
                gal_x4_mp.deflections_from_grid(grid=np.array([[51.0, 51.0]]))[0, 1],
                1e-5,
            )

    class TestMassProfileGeometry:
        def test__extracts_centres_correctly__ignores_mass_sheets(self):

            galaxy = ag.Galaxy(redshift=0.5)

            assert galaxy.mass_profile_centres == []

            galaxy = ag.Galaxy(
                redshift=0.5, mp_0=ag.mp.EllipticalMassProfile(centre=(0.0, 1.0))
            )

            assert galaxy.mass_profile_centres.in_list == [[(0.0, 1.0)]]
            assert galaxy.mass_profile_centres.as_dict["mp_0"] == [(0.0, 1.0)]

            galaxy = ag.Galaxy(
                redshift=0.5,
                mp_0=ag.mp.EllipticalMassProfile(centre=(0.0, 1.0)),
                mp_1=ag.mp.EllipticalMassProfile(centre=(2.0, 3.0)),
                mp_2=ag.mp.EllipticalMassProfile(centre=(4.0, 5.0)),
            )

            assert galaxy.mass_profile_centres.in_list == [
                [(0.0, 1.0)],
                [(2.0, 3.0)],
                [(4.0, 5.0)],
            ]
            assert galaxy.mass_profile_centres.as_dict["mp_0"] == [(0.0, 1.0)]
            assert galaxy.mass_profile_centres.as_dict["mp_1"] == [(2.0, 3.0)]
            assert galaxy.mass_profile_centres.as_dict["mp_2"] == [(4.0, 5.0)]

            galaxy = ag.Galaxy(
                redshift=0.5,
                mp_0=ag.mp.EllipticalMassProfile(centre=(0.0, 1.0)),
                mp_1=ag.mp.EllipticalMassProfile(centre=(2.0, 3.0)),
                lp_0=ag.lp.EllipticalLightProfile(centre=(-1.0, -2.0)),
                mp_2=ag.mp.EllipticalMassProfile(centre=(4.0, 5.0)),
            )

            assert galaxy.mass_profile_centres.in_list == [
                [(0.0, 1.0)],
                [(2.0, 3.0)],
                [(4.0, 5.0)],
            ]
            assert galaxy.mass_profile_centres.as_dict["mp_0"] == [(0.0, 1.0)]
            assert galaxy.mass_profile_centres.as_dict["mp_1"] == [(2.0, 3.0)]
            assert galaxy.mass_profile_centres.as_dict["mp_2"] == [(4.0, 5.0)]

            galaxy = ag.Galaxy(
                redshift=0.5,
                mp_0=ag.mp.EllipticalMassProfile(centre=(0.0, 1.0)),
                mp_1=ag.mp.EllipticalMassProfile(centre=(2.0, 3.0)),
                lp_0=ag.lp.EllipticalLightProfile(centre=(-1.0, -2.0)),
                mp_2=ag.mp.EllipticalMassProfile(centre=(4.0, 5.0)),
                sheet=ag.mp.MassSheet(centre=(10.0, 10.0)),
            )

            assert galaxy.mass_profile_centres.in_list == [
                [(0.0, 1.0)],
                [(2.0, 3.0)],
                [(4.0, 5.0)],
            ]
            assert galaxy.mass_profile_centres.as_dict["mp_0"] == [(0.0, 1.0)]
            assert galaxy.mass_profile_centres.as_dict["mp_1"] == [(2.0, 3.0)]
            assert galaxy.mass_profile_centres.as_dict["mp_2"] == [(4.0, 5.0)]

        def test__extracts_axis_ratio_correctly(self):

            galaxy = ag.Galaxy(redshift=0.5)

            assert galaxy.mass_profile_axis_ratios == []

            galaxy = ag.Galaxy(
                redshift=0.5,
                mp_0=ag.mp.EllipticalMassProfile(elliptical_comps=(0.0, 0.05263)),
            )

            assert galaxy.mass_profile_axis_ratios.in_list[0][0] == pytest.approx(
                0.9, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.as_dict["mp_0"][0] == pytest.approx(
                0.9, 1.0e-4
            )

            galaxy = ag.Galaxy(
                redshift=0.5,
                mp_0=ag.mp.EllipticalMassProfile(elliptical_comps=(0.0, 0.05263)),
                mp_1=ag.mp.EllipticalMassProfile(elliptical_comps=(0.0, 0.111111)),
                mp_2=ag.mp.EllipticalMassProfile(elliptical_comps=(0.0, 0.176470)),
            )

            assert galaxy.mass_profile_axis_ratios.in_list[0][0] == pytest.approx(
                0.9, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.in_list[1][0] == pytest.approx(
                0.8, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.in_list[2][0] == pytest.approx(
                0.7, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.as_dict["mp_0"][0] == pytest.approx(
                0.9, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.as_dict["mp_1"][0] == pytest.approx(
                0.8, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.as_dict["mp_2"][0] == pytest.approx(
                0.7, 1.0e-4
            )

            galaxy = ag.Galaxy(
                redshift=0.5,
                mp_0=ag.mp.EllipticalMassProfile(elliptical_comps=(0.0, 0.05263)),
                mp_1=ag.mp.EllipticalMassProfile(elliptical_comps=(0.0, 0.111111)),
                lp_0=ag.lp.EllipticalLightProfile(elliptical_comps=(0.0, 0.8181818)),
                mp_2=ag.mp.EllipticalMassProfile(elliptical_comps=(0.0, 0.176470)),
            )

            assert galaxy.mass_profile_axis_ratios.in_list[0][0] == pytest.approx(
                0.9, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.in_list[1][0] == pytest.approx(
                0.8, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.in_list[2][0] == pytest.approx(
                0.7, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.as_dict["mp_0"][0] == pytest.approx(
                0.9, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.as_dict["mp_1"][0] == pytest.approx(
                0.8, 1.0e-4
            )
            assert galaxy.mass_profile_axis_ratios.as_dict["mp_2"][0] == pytest.approx(
                0.7, 1.0e-4
            )

        def test__extracts_phis_correctly(self):

            galaxy = ag.Galaxy(redshift=0.5)

            assert galaxy.mass_profile_phis == []

            galaxy = ag.Galaxy(
                redshift=0.5,
                mp_0=ag.mp.EllipticalMassProfile.from_axis_ratio_and_phi(
                    axis_ratio=0.5, phi=0.9
                ),
            )

            assert galaxy.mass_profile_phis.in_list == [[0.9]]
            assert galaxy.mass_profile_phis.as_dict["mp_0"][0] == pytest.approx(
                0.9, 1.0e-4
            )

            galaxy = ag.Galaxy(
                redshift=0.5,
                mp_0=ag.mp.EllipticalMassProfile.from_axis_ratio_and_phi(
                    axis_ratio=0.5, phi=0.9
                ),
                mp_1=ag.mp.EllipticalMassProfile.from_axis_ratio_and_phi(
                    axis_ratio=0.5, phi=0.8
                ),
                mp_2=ag.mp.EllipticalMassProfile.from_axis_ratio_and_phi(
                    axis_ratio=0.5, phi=0.7
                ),
            )

            assert galaxy.mass_profile_phis.in_list[0][0] == pytest.approx(0.9, 1.0e-4)
            assert galaxy.mass_profile_phis.in_list[1][0] == pytest.approx(0.8, 1.0e-4)
            assert galaxy.mass_profile_phis.in_list[2][0] == pytest.approx(0.7, 1.0e-4)
            assert galaxy.mass_profile_phis.as_dict["mp_0"][0] == pytest.approx(
                0.9, 1.0e-4
            )
            assert galaxy.mass_profile_phis.as_dict["mp_1"][0] == pytest.approx(
                0.8, 1.0e-4
            )
            assert galaxy.mass_profile_phis.as_dict["mp_2"][0] == pytest.approx(
                0.7, 1.0e-4
            )

            galaxy = ag.Galaxy(
                redshift=0.5,
                mp_0=ag.mp.EllipticalMassProfile.from_axis_ratio_and_phi(
                    axis_ratio=0.5, phi=0.9
                ),
                mp_1=ag.mp.EllipticalMassProfile.from_axis_ratio_and_phi(
                    axis_ratio=0.5, phi=0.8
                ),
                lp_0=ag.lp.EllipticalLightProfile.from_axis_ratio_and_phi(
                    axis_ratio=0.5, phi=0.1
                ),
                mp_2=ag.mp.EllipticalMassProfile.from_axis_ratio_and_phi(
                    axis_ratio=0.5, phi=0.7
                ),
            )

            assert galaxy.mass_profile_phis.in_list[0][0] == pytest.approx(0.9, 1.0e-4)
            assert galaxy.mass_profile_phis.in_list[1][0] == pytest.approx(0.8, 1.0e-4)
            assert galaxy.mass_profile_phis.in_list[2][0] == pytest.approx(0.7, 1.0e-4)
            assert galaxy.mass_profile_phis.as_dict["mp_0"][0] == pytest.approx(
                0.9, 1.0e-4
            )
            assert galaxy.mass_profile_phis.as_dict["mp_1"][0] == pytest.approx(
                0.8, 1.0e-4
            )
            assert galaxy.mass_profile_phis.as_dict["mp_2"][0] == pytest.approx(
                0.7, 1.0e-4
            )

    class TestLensingObject:
        def test__correct_einstein_mass_caclulated_for_multiple_mass_profiles__means_all_innherited_methods_work(
            self,
        ):

            sis_0 = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=2.0)

            sis_1 = ag.mp.SphericalIsothermal(centre=(0.0, 0.0), einstein_radius=1.0)

            galaxy = ag.Galaxy(mass_profile_0=sis_0, mass_profile_1=sis_1, redshift=0.5)

            assert galaxy.einstein_mass_in_units(unit_mass="angular") == pytest.approx(
                np.pi * 3.0 ** 2.0, 1.0e-1
            )


class TestMassAndLightProfiles:
    def test_single_profile(self, lmp_0):
        gal_x1_lmp = ag.Galaxy(redshift=0.5, profile=lmp_0)

        assert 1 == len(gal_x1_lmp.light_profiles)
        assert 1 == len(gal_x1_lmp.mass_profiles)

        assert gal_x1_lmp.mass_profiles[0] == lmp_0
        assert gal_x1_lmp.light_profiles[0] == lmp_0

    def test_multiple_profile(self, lmp_0, lp_0, mp_0):
        gal_multi_profiles = ag.Galaxy(
            redshift=0.5, profile=lmp_0, light=lp_0, sie=mp_0
        )

        assert 2 == len(gal_multi_profiles.light_profiles)
        assert 2 == len(gal_multi_profiles.mass_profiles)


class TestSummarizeInUnits:
    def test__galaxy_with_two_light_and_mass_profiles(self, lp_0, lp_1, mp_0, mp_1):

        test_path = "{}/config/summary".format(
            os.path.dirname(os.path.realpath(__file__))
        )
        conf.instance = conf.Config(config_path=test_path)

        gal_summarize = ag.Galaxy(
            redshift=0.5,
            light_profile_0=lp_0,
            light_profile_1=lp_1,
            mass_profile_0=mp_0,
            mass_profile_1=mp_1,
        )

        summary_text = gal_summarize.summarize_in_units(
            radii=[ag.dim.Length(10.0), ag.dim.Length(500.0)],
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
            == "einstein_radius                                   2.99 arcsec"
        )
        i += 1
        assert (
            summary_text[i]
            == "einstein_mass                                     2.8177e+01 angular"
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
            == "einstein_mass                                     3.1308e+00 angular"
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
            == "einstein_mass                                     1.2523e+01 angular"
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


class TestHyperGalaxy:
    class TestContributionMaps:
        def test__model_image_all_1s__factor_is_0__contributions_all_1s(self):
            hyper_image = np.ones((3,))

            hyp = ag.HyperGalaxy(contribution_factor=0.0)
            contribution_map = hyp.contribution_map_from_hyper_images(
                hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image
            )

            assert (contribution_map == np.ones((3,))).all()

        def test__different_values__factor_is_1__contributions_are_value_divided_by_factor_and_max(
            self
        ):
            hyper_image = np.array([0.5, 1.0, 1.5])

            hyp = ag.HyperGalaxy(contribution_factor=1.0)
            contribution_map = hyp.contribution_map_from_hyper_images(
                hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image
            )

            assert (
                contribution_map
                == np.array([(0.5 / 1.5) / (1.5 / 2.5), (1.0 / 2.0) / (1.5 / 2.5), 1.0])
            ).all()

        def test__galaxy_contribution_map_property(self):

            hyper_image = np.ones((3,))

            hyp = ag.HyperGalaxy(contribution_factor=0.0)

            galaxy = ag.Galaxy(
                redshift=0.5,
                hyper_galaxy=hyp,
                hyper_galaxy_image=hyper_image,
                hyper_model_image=hyper_image,
            )

            contribution_map = hyp.contribution_map_from_hyper_images(
                hyper_model_image=hyper_image, hyper_galaxy_image=hyper_image
            )

            assert (contribution_map == galaxy.contribution_map).all()

    class TestHyperNoiseMap:
        def test__contribution_all_1s__noise_factor_2__noise_adds_double(self):
            noise_map = np.array([1.0, 2.0, 3.0])
            contribution_map = np.ones((3, 1))

            hyper_galaxy = ag.HyperGalaxy(
                contribution_factor=0.0, noise_factor=2.0, noise_power=1.0
            )

            hyper_noise_map = hyper_galaxy.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map
            )

            assert (hyper_noise_map == np.array([2.0, 4.0, 6.0])).all()

        def test__same_as_above_but_contributions_vary(self):
            noise_map = np.array([1.0, 2.0, 3.0])
            contribution_map = np.array([[0.0, 0.5, 1.0]])

            hyper_galaxy = ag.HyperGalaxy(
                contribution_factor=0.0, noise_factor=2.0, noise_power=1.0
            )

            hyper_noise_map = hyper_galaxy.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map
            )

            assert (hyper_noise_map == np.array([0.0, 2.0, 6.0])).all()

        def test__same_as_above_but_change_noise_scale_terms(self):
            noise_map = np.array([1.0, 2.0, 3.0])
            contribution_map = np.array([[0.0, 0.5, 1.0]])

            hyper_galaxy = ag.HyperGalaxy(
                contribution_factor=0.0, noise_factor=2.0, noise_power=2.0
            )

            hyper_noise_map = hyper_galaxy.hyper_noise_map_from_contribution_map(
                noise_map=noise_map, contribution_map=contribution_map
            )

            assert (hyper_noise_map == np.array([0.0, 2.0, 18.0])).all()


class TestBooleanProperties:
    def test_has_profile(self):
        assert ag.Galaxy(redshift=0.5).has_profile is False
        assert (
            ag.Galaxy(redshift=0.5, light_profile=ag.lp.LightProfile()).has_profile
            is True
        )
        assert (
            ag.Galaxy(redshift=0.5, mass_profile=ag.mp.MassProfile()).has_profile
            is True
        )

    def test_has_light_profile(self):
        assert ag.Galaxy(redshift=0.5).has_light_profile is False
        assert (
            ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.LightProfile()
            ).has_light_profile
            is True
        )
        assert (
            ag.Galaxy(redshift=0.5, mass_profile=ag.mp.MassProfile()).has_light_profile
            is False
        )

    def test_has_mass_profile(self):
        assert ag.Galaxy(redshift=0.5).has_mass_profile is False
        assert (
            ag.Galaxy(redshift=0.5, light_profile=ag.lp.LightProfile()).has_mass_profile
            is False
        )
        assert (
            ag.Galaxy(redshift=0.5, mass_profile=ag.mp.MassProfile()).has_mass_profile
            is True
        )

    def test_has_only_mass_sheets(self):

        assert ag.Galaxy(redshift=0.5).has_only_mass_sheets is False
        assert (
            ag.Galaxy(
                redshift=0.5, light_profile=ag.lp.LightProfile()
            ).has_only_mass_sheets
            is False
        )
        assert (
            ag.Galaxy(
                redshift=0.5, mass_profile=ag.mp.MassProfile()
            ).has_only_mass_sheets
            is False
        )
        assert (
            ag.Galaxy(
                redshift=0.5, mass_profile=ag.mp.MassProfile(), sheet=ag.mp.MassSheet()
            ).has_only_mass_sheets
            is False
        )
        assert (
            ag.Galaxy(redshift=0.5, sheet=ag.mp.MassSheet()).has_only_mass_sheets
            is True
        )

    def test_has_redshift(self):
        assert ag.Galaxy(redshift=0.1).has_redshift is True

    def test_has_pixelization(self):
        assert ag.Galaxy(redshift=0.5).has_pixelization is False
        assert (
            ag.Galaxy(
                redshift=0.5, pixelization=object(), regularization=object()
            ).has_pixelization
            is True
        )

    def test_has_regularization(self):
        assert ag.Galaxy(redshift=0.5).has_regularization is False
        assert (
            ag.Galaxy(
                redshift=0.5, pixelization=object(), regularization=object()
            ).has_regularization
            is True
        )

    def test_has_hyper_galaxy(self):
        assert ag.Galaxy(redshift=0.5, hyper_galaxy=object()).has_hyper_galaxy is True

    def test__only_pixelization_raises_error(self):
        with pytest.raises(exc.GalaxyException):
            ag.Galaxy(redshift=0.5, pixelization=object())

    def test__only_regularization_raises_error(self):
        with pytest.raises(exc.GalaxyException):
            ag.Galaxy(redshift=0.5, regularization=object())


class TestRegression:
    def test__centre_of_profile_in_right_place(self):

        grid = ag.Grid.uniform(shape_2d=(7, 7), pixel_scales=1.0)

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=ag.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )

        convergence = galaxy.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = galaxy.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = galaxy.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] > 0
        assert deflections.in_2d[2, 4, 0] < 0
        assert deflections.in_2d[1, 4, 1] > 0
        assert deflections.in_2d[1, 3, 1] < 0

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=ag.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )
        convergence = galaxy.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = galaxy.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = galaxy.deflections_from_grid(grid=grid)
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

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
            mass_0=ag.mp.EllipticalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )
        convergence = galaxy.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = galaxy.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = galaxy.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] >= 0
        assert deflections.in_2d[2, 4, 0] <= 0
        assert deflections.in_2d[1, 4, 1] >= 0
        assert deflections.in_2d[1, 3, 1] <= 0

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.SphericalIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        )
        convergence = galaxy.convergence_from_grid(grid=grid)
        max_indexes = np.unravel_index(convergence.in_2d.argmax(), convergence.shape_2d)
        assert max_indexes == (1, 4)

        potential = galaxy.potential_from_grid(grid=grid)
        max_indexes = np.unravel_index(potential.in_2d.argmin(), potential.shape_2d)
        assert max_indexes == (1, 4)

        deflections = galaxy.deflections_from_grid(grid=grid)
        assert deflections.in_2d[1, 4, 0] >= 0
        assert deflections.in_2d[2, 4, 0] <= 0
        assert deflections.in_2d[1, 4, 1] >= 0
        assert deflections.in_2d[1, 3, 1] <= 0


class TestDecorators:
    def test__grid_iterate_in__iterates_array_result_correctly(self, gal_x1_lp):

        mask = ag.Mask.manual(
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

        galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllipticalSersic(intensity=1.0))

        image = galaxy.image_from_grid(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = ag.Grid.from_mask(mask=mask_sub_2)
        image_sub_2 = galaxy.image_from_grid(grid=grid_sub_2).in_1d_binned

        assert (image == image_sub_2).all()

        grid = ag.GridIterate.from_mask(
            mask=mask, fractional_accuracy=0.95, sub_steps=[2, 4, 8]
        )

        galaxy = ag.Galaxy(
            redshift=0.5,
            light=ag.lp.EllipticalSersic(centre=(0.08, 0.08), intensity=1.0),
        )

        image = galaxy.image_from_grid(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = ag.Grid.from_mask(mask=mask_sub_4)
        image_sub_4 = galaxy.image_from_grid(grid=grid_sub_4).in_1d_binned

        assert image[0] == image_sub_4[0]

        mask_sub_8 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)
        grid_sub_8 = ag.Grid.from_mask(mask=mask_sub_8)
        image_sub_8 = galaxy.image_from_grid(grid=grid_sub_8).in_1d_binned

        assert image[4] == image_sub_8[4]

    def test__grid_iterate_in__iterates_grid_result_correctly(self, gal_x1_mp):

        mask = ag.Mask.manual(
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

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllipticalIsothermal(centre=(0.08, 0.08), einstein_radius=1.0),
        )

        deflections = galaxy.deflections_from_grid(grid=grid)

        mask_sub_2 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=2)
        grid_sub_2 = ag.Grid.from_mask(mask=mask_sub_2)
        deflections_sub_2 = galaxy.deflections_from_grid(grid=grid_sub_2).in_1d_binned

        assert (deflections == deflections_sub_2).all()

        grid = ag.GridIterate.from_mask(
            mask=mask, fractional_accuracy=0.99, sub_steps=[2, 4, 8]
        )

        galaxy = ag.Galaxy(
            redshift=0.5,
            mass=ag.mp.EllipticalIsothermal(centre=(0.08, 0.08), einstein_radius=1.0),
        )

        deflections = galaxy.deflections_from_grid(grid=grid)

        mask_sub_4 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=4)
        grid_sub_4 = ag.Grid.from_mask(mask=mask_sub_4)
        deflections_sub_4 = galaxy.deflections_from_grid(grid=grid_sub_4).in_1d_binned

        assert deflections[0, 0] == deflections_sub_4[0, 0]

        mask_sub_8 = mask.mask_new_sub_size_from_mask(mask=mask, sub_size=8)
        grid_sub_8 = ag.Grid.from_mask(mask=mask_sub_8)
        deflections_sub_8 = galaxy.deflections_from_grid(grid=grid_sub_8).in_1d_binned

        assert deflections[4, 0] == deflections_sub_8[4, 0]

    def test__grid_interp_in__interps_based_on_intepolate_config(self):

        # False in interpolate.ini

        mask = ag.Mask.manual(
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

        grid_interp = ag.GridInterpolate.from_mask(mask=mask, pixel_scales_interp=0.1)

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

        image = galaxy.image_from_grid(grid=grid_interp)

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

        convergence = galaxy.convergence_from_grid(grid=grid_interp)

        assert (convergence == convergence_no_interp + convergence_interp).all()

        potential_no_interp = mass_profile.potential_from_grid(grid=grid)

        array_interp = mass_profile_interp.potential_from_grid(
            grid=grid_interp.grid_interp
        )
        potential_interp = grid_interp.interpolated_array_from_array_interp(
            array_interp=array_interp
        )

        galaxy = ag.Galaxy(redshift=0.5, mass=mass_profile_interp, mass_0=mass_profile)

        potential = galaxy.potential_from_grid(grid=grid_interp)

        assert (potential == potential_no_interp + potential_interp).all()

        deflections_no_interp = mass_profile.deflections_from_grid(grid=grid)

        grid_interp_0 = mass_profile_interp.deflections_from_grid(
            grid=grid_interp.grid_interp
        )
        deflections_interp = grid_interp.interpolated_grid_from_grid_interp(
            grid_interp=grid_interp_0
        )

        galaxy = ag.Galaxy(redshift=0.5, mass=mass_profile_interp, mass_0=mass_profile)

        deflections = galaxy.deflections_from_grid(grid=grid_interp)

        assert (deflections == deflections_no_interp + deflections_interp).all()
