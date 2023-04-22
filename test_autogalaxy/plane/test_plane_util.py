import pytest

import autogalaxy as ag

from autogalaxy import exc


def test__plane_image_from(sub_grid_2d_7x7):

    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(intensity=1.0))

    plane_image = ag.util.plane.plane_image_from(
        grid=sub_grid_2d_7x7, galaxies=[galaxy], buffer=0.1
    )

    assert plane_image[0] == pytest.approx(12.5227, 1.0e-4)


def test__ordered_plane_redshifts_from():
    galaxies = [
        ag.Galaxy(redshift=2.0),
        ag.Galaxy(redshift=1.0),
        ag.Galaxy(redshift=0.1),
    ]

    ordered_plane_redshifts = ag.util.plane.ordered_plane_redshifts_from(
        galaxies=galaxies
    )

    assert ordered_plane_redshifts == [0.1, 1.0, 2.0]

    galaxies = [
        ag.Galaxy(redshift=1.0),
        ag.Galaxy(redshift=1.0),
        ag.Galaxy(redshift=0.1),
    ]

    ordered_plane_redshifts = ag.util.plane.ordered_plane_redshifts_from(
        galaxies=galaxies
    )

    assert ordered_plane_redshifts == [0.1, 1.0]

    g0 = ag.Galaxy(redshift=1.0)
    g1 = ag.Galaxy(redshift=1.0)
    g2 = ag.Galaxy(redshift=0.1)
    g3 = ag.Galaxy(redshift=1.05)
    g4 = ag.Galaxy(redshift=0.95)
    g5 = ag.Galaxy(redshift=1.05)

    galaxies = [g0, g1, g2, g3, g4, g5]

    ordered_plane_redshifts = ag.util.plane.ordered_plane_redshifts_from(
        galaxies=galaxies
    )

    assert ordered_plane_redshifts == [0.1, 0.95, 1.0, 1.05]


def test__ordered_plane_redshifts_with_slicing_from():
    ordered_plane_redshifts = ag.util.plane.ordered_plane_redshifts_with_slicing_from(
        lens_redshifts=[1.0],
        source_plane_redshift=3.0,
        planes_between_lenses=[1, 1],
    )

    assert ordered_plane_redshifts == [0.5, 1.0, 2.0]

    ordered_plane_redshifts = ag.util.plane.ordered_plane_redshifts_with_slicing_from(
        lens_redshifts=[1.0],
        source_plane_redshift=2.0,
        planes_between_lenses=[2, 3],
    )

    assert ordered_plane_redshifts == [
        (1.0 / 3.0),
        (2.0 / 3.0),
        1.0,
        1.25,
        1.5,
        1.75,
    ]

    with pytest.raises(exc.PlaneException):
        ag.util.plane.ordered_plane_redshifts_with_slicing_from(
            lens_redshifts=[1.0],
            source_plane_redshift=2.0,
            planes_between_lenses=[2, 3, 1],
        )

    with pytest.raises(exc.PlaneException):
        ag.util.plane.ordered_plane_redshifts_with_slicing_from(
            lens_redshifts=[1.0],
            source_plane_redshift=2.0,
            planes_between_lenses=[2],
        )

    with pytest.raises(exc.PlaneException):
        ag.util.plane.ordered_plane_redshifts_with_slicing_from(
            lens_redshifts=[1.0, 3.0],
            source_plane_redshift=2.0,
            planes_between_lenses=[2],
        )


def test__galaxies_in_redshift_ordered_planes_from():
    galaxies = [
        ag.Galaxy(redshift=2.0),
        ag.Galaxy(redshift=1.0),
        ag.Galaxy(redshift=0.1),
    ]

    ordered_plane_redshifts = [0.1, 1.0, 2.0]

    galaxies_in_redshift_ordered_planes = (
        ag.util.plane.galaxies_in_redshift_ordered_planes_from(
            galaxies=galaxies, plane_redshifts=ordered_plane_redshifts
        )
    )

    assert galaxies_in_redshift_ordered_planes[0][0].redshift == 0.1
    assert galaxies_in_redshift_ordered_planes[1][0].redshift == 1.0
    assert galaxies_in_redshift_ordered_planes[2][0].redshift == 2.0

    galaxies = [
        ag.Galaxy(redshift=1.0),
        ag.Galaxy(redshift=1.0),
        ag.Galaxy(redshift=0.1),
    ]

    ordered_plane_redshifts = [0.1, 1.0]

    galaxies_in_redshift_ordered_planes = (
        ag.util.plane.galaxies_in_redshift_ordered_planes_from(
            galaxies=galaxies, plane_redshifts=ordered_plane_redshifts
        )
    )

    assert galaxies_in_redshift_ordered_planes[0][0].redshift == 0.1
    assert galaxies_in_redshift_ordered_planes[1][0].redshift == 1.0
    assert galaxies_in_redshift_ordered_planes[1][1].redshift == 1.0

    g0 = ag.Galaxy(redshift=1.0)
    g1 = ag.Galaxy(redshift=1.0)
    g2 = ag.Galaxy(redshift=0.1)
    g3 = ag.Galaxy(redshift=1.05)
    g4 = ag.Galaxy(redshift=0.95)
    g5 = ag.Galaxy(redshift=1.05)

    galaxies = [g0, g1, g2, g3, g4, g5]

    ordered_plane_redshifts = [0.1, 0.95, 1.0, 1.05]

    galaxies_in_redshift_ordered_planes = (
        ag.util.plane.galaxies_in_redshift_ordered_planes_from(
            galaxies=galaxies, plane_redshifts=ordered_plane_redshifts
        )
    )

    assert galaxies_in_redshift_ordered_planes[0][0].redshift == 0.1
    assert galaxies_in_redshift_ordered_planes[1][0].redshift == 0.95
    assert galaxies_in_redshift_ordered_planes[2][0].redshift == 1.0
    assert galaxies_in_redshift_ordered_planes[2][1].redshift == 1.0
    assert galaxies_in_redshift_ordered_planes[3][0].redshift == 1.05
    assert galaxies_in_redshift_ordered_planes[3][1].redshift == 1.05

    assert galaxies_in_redshift_ordered_planes[0] == [g2]
    assert galaxies_in_redshift_ordered_planes[1] == [g4]
    assert galaxies_in_redshift_ordered_planes[2] == [g0, g1]
    assert galaxies_in_redshift_ordered_planes[3] == [g3, g5]


def test___galaxies_in_redshift_ordered_planes_from__galaxy_redshifts_dont_match_so_go_to_nearest_plane():
    ordered_plane_redshifts = [0.5, 1.0, 2.0, 3.0]

    galaxies = [
        ag.Galaxy(redshift=0.2),
        ag.Galaxy(redshift=0.4),
        ag.Galaxy(redshift=0.8),
        ag.Galaxy(redshift=1.2),
        ag.Galaxy(redshift=2.9),
    ]

    galaxies_in_redshift_ordered_planes = (
        ag.util.plane.galaxies_in_redshift_ordered_planes_from(
            galaxies=galaxies, plane_redshifts=ordered_plane_redshifts
        )
    )

    assert galaxies_in_redshift_ordered_planes[0][0].redshift == 0.2
    assert galaxies_in_redshift_ordered_planes[0][1].redshift == 0.4
    assert galaxies_in_redshift_ordered_planes[1][0].redshift == 0.8
    assert galaxies_in_redshift_ordered_planes[1][1].redshift == 1.2
    assert galaxies_in_redshift_ordered_planes[2] == []
    assert galaxies_in_redshift_ordered_planes[3][0].redshift == 2.9

    ordered_plane_redshifts = [(1.0 / 3.0), (2.0 / 3.0), 1.0, 1.25, 1.5, 1.75, 2.0]

    galaxies = [
        ag.Galaxy(redshift=0.1),
        ag.Galaxy(redshift=0.2),
        ag.Galaxy(redshift=1.25),
        ag.Galaxy(redshift=1.35),
        ag.Galaxy(redshift=1.45),
        ag.Galaxy(redshift=1.55),
        ag.Galaxy(redshift=1.9),
    ]

    galaxies_in_redshift_ordered_planes = (
        ag.util.plane.galaxies_in_redshift_ordered_planes_from(
            galaxies=galaxies, plane_redshifts=ordered_plane_redshifts
        )
    )

    assert galaxies_in_redshift_ordered_planes[0][0].redshift == 0.1
    assert galaxies_in_redshift_ordered_planes[0][1].redshift == 0.2
    assert galaxies_in_redshift_ordered_planes[1] == []
    assert galaxies_in_redshift_ordered_planes[2] == []
    assert galaxies_in_redshift_ordered_planes[3][0].redshift == 1.25
    assert galaxies_in_redshift_ordered_planes[3][1].redshift == 1.35
    assert galaxies_in_redshift_ordered_planes[4][0].redshift == 1.45
    assert galaxies_in_redshift_ordered_planes[4][1].redshift == 1.55
    assert galaxies_in_redshift_ordered_planes[6][0].redshift == 1.9
