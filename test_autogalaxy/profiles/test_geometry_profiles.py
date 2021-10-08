from __future__ import division, print_function

from os import path

import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.profiles import geometry_profiles

directory = path.dirname(path.realpath(__file__))


class TestEllProfile:
    def test__cos_and_sin_from_x(self):

        elliptical_profile = geometry_profiles.EllProfile(
            centre=(1.0, 1.0), elliptical_comps=(0.0, 0.0)
        )

        cos_phi, sin_phi = elliptical_profile.cos_and_sin_to_x_axis()

        assert cos_phi == 1.0
        assert sin_phi == 0.0

        elliptical_profile = geometry_profiles.EllProfile.from_axis_ratio_and_phi(
            centre=(1, 1), axis_ratio=0.1, angle=45.0
        )

        cos_phi, sin_phi = elliptical_profile.cos_and_sin_to_x_axis()

        assert cos_phi == pytest.approx(0.707, 1e-3)
        assert sin_phi == pytest.approx(0.707, 1e-3)

        elliptical_profile = geometry_profiles.EllProfile.from_axis_ratio_and_phi(
            centre=(1, 1), axis_ratio=0.1, angle=60.0
        )

        cos_phi, sin_phi = elliptical_profile.cos_and_sin_to_x_axis()

        assert cos_phi == pytest.approx(0.5, 1e-3)
        assert sin_phi == pytest.approx(0.866, 1e-3)

        elliptical_profile = geometry_profiles.EllProfile.from_axis_ratio_and_phi(
            centre=(1, 1), axis_ratio=0.1, angle=225.0
        )

        cos_phi, sin_phi = elliptical_profile.cos_and_sin_to_x_axis()

        assert cos_phi == pytest.approx(0.707, 1e-3)
        assert sin_phi == pytest.approx(0.707, 1e-3)

    def test__transform_grid_to_and_from_reference_frame(self):

        elliptical_profile = geometry_profiles.EllProfile(elliptical_comps=(0.0, 0.0))

        transformed_grid = elliptical_profile.transform_grid_to_reference_frame(
            grid=np.array([[1.0, 1.0]])
        )

        assert transformed_grid == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)

        transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(
            transformed_grid
        )

        assert transformed_back_grid == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)

        transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(
            transformed_grid
        )

        assert transformed_back_grid == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)

        elliptical_profile = geometry_profiles.EllProfile.from_axis_ratio_and_phi(
            axis_ratio=0.1, angle=90.0, centre=(2.0, 3.0)
        )

        transformed_grid = elliptical_profile.transform_grid_to_reference_frame(
            grid=np.array([[3.0, 4.0]])
        )

        assert transformed_grid == pytest.approx(np.array([[-1.0, 1.0]]), 1e-3)

        transformed_back_grid = elliptical_profile.transform_grid_from_reference_frame(
            transformed_grid
        )

        assert transformed_back_grid == pytest.approx(np.array([[3.0, 4.0]]), 1e-3)

        elliptical_profile = geometry_profiles.EllProfile.from_axis_ratio_and_phi(
            axis_ratio=0.1, angle=45.0
        )

        grid_original = np.array([[5.2221, 2.6565]])

        grid_elliptical = elliptical_profile.transform_grid_to_reference_frame(
            grid_original
        )

        transformed_grid = elliptical_profile.transform_grid_from_reference_frame(
            grid_elliptical
        )

        assert transformed_grid == pytest.approx(grid_original, 1e-5)

    def test__transform_grids_with_mapped_centres(self,):
        elliptical_profile1 = geometry_profiles.EllProfile(
            elliptical_comps=(0.0, 0.0), centre=(0, 0)
        )
        grid1 = elliptical_profile1.transform_grid_to_reference_frame(
            grid=np.array([[1.0, 1.0]])
        )

        elliptical_profile2 = geometry_profiles.EllProfile(
            elliptical_comps=(0.0, 0.0), centre=(-1, -1)
        )
        grid2 = elliptical_profile2.transform_grid_to_reference_frame(
            grid=np.array([[0.0, 0.0]])
        )

        assert (grid1 == grid2).all()

        elliptical_profile1 = geometry_profiles.EllProfile.from_axis_ratio_and_phi(
            axis_ratio=0.1, angle=55.0, centre=(0, 0)
        )
        grid1 = elliptical_profile1.transform_grid_to_reference_frame(
            grid=np.array([[1.0, 1.0]])
        )

        elliptical_profile2 = geometry_profiles.EllProfile.from_axis_ratio_and_phi(
            axis_ratio=0.1, angle=55.0, centre=(-1, -1)
        )
        grid2 = elliptical_profile2.transform_grid_to_reference_frame(
            grid=np.array([[0.0, 0.0]])
        )

        assert (grid1 == grid2).all()

        elliptical_profile1 = geometry_profiles.EllProfile.from_axis_ratio_and_phi(
            axis_ratio=0.1, angle=55.0, centre=(1, 1)
        )
        grid1 = elliptical_profile1.transform_grid_to_reference_frame(
            grid=np.array([[1.0, 1.0]])
        )

        elliptical_profile2 = geometry_profiles.EllProfile.from_axis_ratio_and_phi(
            axis_ratio=0.1, angle=55.0, centre=(-1, -1)
        )
        grid2 = elliptical_profile2.transform_grid_to_reference_frame(
            grid=np.array([[-1.0, -1.0]])
        )

        assert (grid1 == grid2).all()

    def test__grid_to_eccentric_radii(self,):
        elliptical_profile = geometry_profiles.EllProfile(elliptical_comps=(0.0, 0.0))

        eccentric_radius = elliptical_profile.grid_to_eccentric_radii(
            grid=np.array([[1.0, 1.0]])
        )

        assert eccentric_radius == pytest.approx(2.0 ** 0.5, 1e-3)

        eccentric_radius = elliptical_profile.grid_to_eccentric_radii(
            grid=np.array([[1.0, 1.0]])
        )

        assert eccentric_radius == pytest.approx(2.0 ** 0.5, 1e-3)

        # eccentric_radius = sqrt(axis_ratio) * sqrt(  x**2 +   (y**2 / axis_ratio**2))
        # eccentric_radius =        sqrt(0.5) * sqrt(1.0**2 + (1.0**2 /        0.5**2))
        # eccentric radius =        sqrt(0.5) * sqrt( 5 ) = 1.58113

        elliptical_profile = geometry_profiles.EllProfile.from_axis_ratio_and_phi(
            axis_ratio=0.5, angle=0.0
        )

        eccentric_radius = elliptical_profile.grid_to_eccentric_radii(
            np.array([[1.0, 1.0]])
        )

        assert eccentric_radius == pytest.approx(1.58113, 1e-3)


class TestSphProfile:
    def test__transform_to_reference_frame__coordinate_shifts_using_centre(self):

        spherical_profile = geometry_profiles.SphProfile(centre=(0.0, 0.0))

        transformed_grid = spherical_profile.transform_grid_from_reference_frame(
            np.array([[1.0, 1.0]])
        )

        assert (transformed_grid == np.array([[1.0, 1.0]])).all()

        spherical_profile1 = geometry_profiles.SphProfile(centre=(0, 0))

        grid1 = spherical_profile1.transform_grid_to_reference_frame(
            grid=np.array([[1.0, 1.0]])
        )

        spherical_profile2 = geometry_profiles.SphProfile(centre=(-1, -1))
        grid2 = spherical_profile2.transform_grid_to_reference_frame(
            grid=np.array([[0.0, 0.0]])
        )

        assert (grid1 == grid2).all()

        spherical_profile1 = geometry_profiles.SphProfile(centre=(1, 1))
        grid1 = spherical_profile1.transform_grid_to_reference_frame(
            grid=np.array([[1.0, 1.0]])
        )

        spherical_profile2 = geometry_profiles.SphProfile(centre=(-1, -1))
        grid2 = spherical_profile2.transform_grid_to_reference_frame(
            grid=np.array([[-1.0, -1.0]])
        )

        assert (grid1 == grid2).all()

    def test__transform_to_and_from_reference_frame(self,):

        spherical_profile = geometry_profiles.SphProfile(centre=(0.0, 0.0))

        grid = np.array([[1.0, 1.0]])

        transformed_grid = spherical_profile.transform_grid_to_reference_frame(grid)

        assert transformed_grid == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)

        spherical_profile = geometry_profiles.SphProfile(centre=(0.0, 0.0))

        grid_original = np.array([[5.2221, 2.6565]])

        grid_spherical = spherical_profile.transform_grid_to_reference_frame(
            grid_original
        )

        transformed_grid = spherical_profile.transform_grid_from_reference_frame(
            grid_spherical
        )

        assert transformed_grid == pytest.approx(grid_original, 1e-5)
