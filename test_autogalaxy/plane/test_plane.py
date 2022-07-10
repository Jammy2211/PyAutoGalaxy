import numpy as np
import pytest
from skimage import measure

import autogalaxy as ag

from autoarray.inversion.pixelizations.abstract import AbstractPixelization
from autoarray.inversion.regularization.abstract import AbstractRegularization

from autogalaxy import exc
from autogalaxy.plane import plane


def critical_curve_via_magnification_via_plane_from(plane, grid):

    magnification = plane.magnification_2d_from(grid=grid)

    inverse_magnification = 1 / magnification

    critical_curves_indices = measure.find_contours(inverse_magnification.native, 0)

    no_critical_curves = len(critical_curves_indices)
    contours = []
    critical_curves = []

    for jj in np.arange(no_critical_curves):
        contours.append(critical_curves_indices[jj])
        contour_x, contour_y = contours[jj].T
        pixel_coord = np.stack((contour_x, contour_y), axis=-1)

        critical_curve = grid.mask.grid_scaled_for_marching_squares_from(
            grid_pixels_1d=pixel_coord, shape_native=magnification.sub_shape_native
        )

        critical_curve = np.array(grid=critical_curve)

        critical_curves.append(critical_curve)

    return critical_curves


def caustics_via_magnification_via_plane_from(plane, grid):
    caustics = []

    critical_curves = critical_curve_via_magnification_via_plane_from(
        plane=plane, grid=grid
    )

    for i in range(len(critical_curves)):
        critical_curve = critical_curves[i]

        deflections_1d = plane.deflections_yx_2d_from(grid=critical_curve)

        caustic = critical_curve - deflections_1d

        caustics.append(caustic)

    return caustics


### Has Attributes ###


def test__has():

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)
    assert plane.has(cls=ag.lp.LightProfile) is False

    plane = ag.Plane(
        galaxies=[ag.Galaxy(redshift=0.5, light_profile=ag.lp.LightProfile())],
        redshift=None,
    )
    assert plane.has(cls=ag.lp.LightProfile) is True

    plane = ag.Plane(
        galaxies=[
            ag.Galaxy(redshift=0.5, light_profile=ag.lp.LightProfile()),
            ag.Galaxy(redshift=0.5),
        ],
        redshift=None,
    )
    assert plane.has(cls=ag.lp.LightProfile) is True


### Attribute Lists ###


def test__cls_list_from():

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

    assert plane.cls_list_from(cls=ag.mp.MassProfile) == []

    sis_0 = ag.mp.SphIsothermal(einstein_radius=1.0)
    sis_1 = ag.mp.SphIsothermal(einstein_radius=2.0)
    sis_2 = ag.mp.SphIsothermal(einstein_radius=3.0)

    plane = ag.Plane(
        galaxies=[ag.Galaxy(redshift=0.5, mass_profile=sis_0)], redshift=None
    )
    assert plane.cls_list_from(cls=ag.mp.MassProfile) == [sis_0]

    plane = ag.Plane(
        galaxies=[
            ag.Galaxy(redshift=0.5, mass_profile_0=sis_0, mass_profile_1=sis_1),
            ag.Galaxy(redshift=0.5, mass_profile_0=sis_2, mass_profile_1=sis_1),
        ],
        redshift=None,
    )
    assert plane.cls_list_from(cls=ag.mp.MassProfile) == [sis_0, sis_1, sis_2, sis_1]

    galaxy_pix = ag.Galaxy(
        redshift=0.5,
        pixelization=ag.m.MockPixelization(mapper=1),
        regularization=ag.m.MockRegularization(),
    )

    plane = ag.Plane(galaxies=[galaxy_pix], redshift=None)

    assert plane.cls_list_from(cls=ag.pix.Pixelization)[0].mapper == 1

    galaxy_pix_0 = ag.Galaxy(
        redshift=0.5,
        pixelization=ag.m.MockPixelization(mapper=1),
        regularization=ag.m.MockRegularization(),
    )
    galaxy_pix_1 = ag.Galaxy(
        redshift=0.5,
        pixelization=ag.m.MockPixelization(mapper=2),
        regularization=ag.m.MockRegularization(),
    )

    plane = ag.Plane(galaxies=[galaxy_pix_0, galaxy_pix_1], redshift=None)

    assert plane.cls_list_from(cls=ag.pix.Pixelization)[0].mapper == 1
    assert plane.cls_list_from(cls=ag.pix.Pixelization)[1].mapper == 2

    galaxy_no_pix = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(galaxies=[galaxy_no_pix], redshift=None)

    assert plane.cls_list_from(cls=ag.pix.Pixelization) == []


def test__hyper_galaxy_image_list():
    galaxy_pix = ag.Galaxy(
        redshift=0.5,
        pixelization=AbstractPixelization(),
        regularization=AbstractRegularization(),
    )

    plane = ag.Plane(galaxies=[galaxy_pix], redshift=None)
    assert plane.hyper_galaxies_with_pixelization_image_list[0] is None

    galaxy_pix = ag.Galaxy(
        redshift=0.5,
        pixelization=AbstractPixelization(),
        regularization=AbstractRegularization(),
        hyper_galaxy_image=1,
    )

    plane = ag.Plane(galaxies=[galaxy_pix, ag.Galaxy(redshift=0.5)], redshift=None)
    assert plane.hyper_galaxies_with_pixelization_image_list[0] == 1

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

    assert plane.hyper_galaxies_with_pixelization_image_list == []


### Light Profile Quantities ###


def test__image_2d_from(sub_grid_2d_7x7, gal_x1_lp):
    light_profile = gal_x1_lp.cls_list_from(cls=ag.lp.LightProfile)[0]

    lp_image = light_profile.image_2d_from(grid=sub_grid_2d_7x7)

    # Perform sub gridding average manually
    lp_image_pixel_0 = (lp_image[0] + lp_image[1] + lp_image[2] + lp_image[3]) / 4
    lp_image_pixel_1 = (lp_image[4] + lp_image[5] + lp_image[6] + lp_image[7]) / 4

    plane = ag.Plane(galaxies=[gal_x1_lp], redshift=None)

    image = plane.image_2d_from(grid=sub_grid_2d_7x7)

    assert (image.binned[0] == lp_image_pixel_0).all()
    assert (image.binned[1] == lp_image_pixel_1).all()
    assert (image == lp_image).all()

    galaxy_image = gal_x1_lp.image_2d_from(grid=sub_grid_2d_7x7)

    plane = ag.Plane(galaxies=[gal_x1_lp], redshift=None)

    image = plane.image_2d_from(grid=sub_grid_2d_7x7)

    assert image == pytest.approx(galaxy_image, 1.0e-4)

    # Overwrite one value so intensity in each pixel is different
    sub_grid_2d_7x7[5] = np.array([2.0, 2.0])

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=2.0))

    g0_image = g0.image_2d_from(grid=sub_grid_2d_7x7)

    g1_image = g1.image_2d_from(grid=sub_grid_2d_7x7)

    plane = ag.Plane(galaxies=[g0, g1], redshift=None)

    image = plane.image_2d_from(grid=sub_grid_2d_7x7)

    assert image == pytest.approx(g0_image + g1_image, 1.0e-4)

    plane = ag.Plane(galaxies=[], redshift=0.5)

    image = plane.image_2d_from(grid=sub_grid_2d_7x7)

    assert image.shape_native == (7, 7)
    assert (image[0] == 0.0).all()
    assert (image[1] == 0.0).all()


def test__image_2d_list_from(sub_grid_2d_7x7):
    # Overwrite one value so intensity in each pixel is different
    sub_grid_2d_7x7[5] = np.array([2.0, 2.0])

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=2.0))

    lp0 = g0.cls_list_from(cls=ag.lp.LightProfile)[0]
    lp1 = g1.cls_list_from(cls=ag.lp.LightProfile)[0]

    lp0_image = lp0.image_2d_from(grid=sub_grid_2d_7x7)
    lp1_image = lp1.image_2d_from(grid=sub_grid_2d_7x7)

    # Perform sub gridding average manually
    lp0_image_pixel_0 = (lp0_image[0] + lp0_image[1] + lp0_image[2] + lp0_image[3]) / 4
    lp0_image_pixel_1 = (lp0_image[4] + lp0_image[5] + lp0_image[6] + lp0_image[7]) / 4
    lp1_image_pixel_0 = (lp1_image[0] + lp1_image[1] + lp1_image[2] + lp1_image[3]) / 4
    lp1_image_pixel_1 = (lp1_image[4] + lp1_image[5] + lp1_image[6] + lp1_image[7]) / 4

    plane = ag.Plane(galaxies=[g0, g1], redshift=None)

    image = plane.image_2d_from(grid=sub_grid_2d_7x7)

    assert image.binned[0] == pytest.approx(
        lp0_image_pixel_0 + lp1_image_pixel_0, 1.0e-4
    )
    assert image.binned[1] == pytest.approx(
        lp0_image_pixel_1 + lp1_image_pixel_1, 1.0e-4
    )

    image_of_galaxies = plane.image_2d_list_from(grid=sub_grid_2d_7x7)

    assert image_of_galaxies[0].binned[0] == lp0_image_pixel_0
    assert image_of_galaxies[0].binned[1] == lp0_image_pixel_1
    assert image_of_galaxies[1].binned[0] == lp1_image_pixel_0
    assert image_of_galaxies[1].binned[1] == lp1_image_pixel_1


def test__image_2d_from__operated_only_input(sub_grid_2d_7x7, lp_0, lp_operated_0):

    image_2d_not_operated = lp_0.image_2d_from(grid=sub_grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=sub_grid_2d_7x7)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)
    galaxy_1 = ag.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = ag.Galaxy(redshift=2.0)

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image_2d = plane.image_2d_from(grid=sub_grid_2d_7x7, operated_only=False)
    assert image_2d == pytest.approx(image_2d_not_operated, 1.0e-4)

    image_2d = plane.image_2d_from(grid=sub_grid_2d_7x7, operated_only=True)
    assert image_2d == pytest.approx(3.0 * image_2d_operated, 1.0e-4)

    image_2d = plane.image_2d_from(grid=sub_grid_2d_7x7, operated_only=None)
    assert image_2d == pytest.approx(
        image_2d_not_operated + 3.0 * image_2d_operated, 1.0e-4
    )


def test__image_2d_list_from__operated_only_input(sub_grid_2d_7x7, lp_0, lp_operated_0):

    image_2d_not_operated = lp_0.image_2d_from(grid=sub_grid_2d_7x7)
    image_2d_operated = lp_operated_0.image_2d_from(grid=sub_grid_2d_7x7)

    galaxy_0 = ag.Galaxy(redshift=0.5, light=lp_0, light_operated=lp_operated_0)
    galaxy_1 = ag.Galaxy(
        redshift=1.0, light_operated_0=lp_operated_0, light_operated_1=lp_operated_0
    )
    galaxy_2 = ag.Galaxy(redshift=2.0)

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1, galaxy_2])

    image_2d_list = plane.image_2d_list_from(grid=sub_grid_2d_7x7, operated_only=False)
    assert image_2d_list[0] == pytest.approx(image_2d_not_operated, 1.0e-4)
    assert image_2d_list[1] == pytest.approx(np.zeros((36)), 1.0e-4)
    assert image_2d_list[2] == pytest.approx(np.zeros((36)), 1.0e-4)

    image_2d_list = plane.image_2d_list_from(grid=sub_grid_2d_7x7, operated_only=True)
    assert image_2d_list[0] == pytest.approx(image_2d_operated, 1.0e-4)
    assert image_2d_list[1] == pytest.approx(2.0 * image_2d_operated, 1.0e-4)
    assert image_2d_list[2] == pytest.approx(np.zeros((36)), 1.0e-4)

    image_2d_list = plane.image_2d_list_from(grid=sub_grid_2d_7x7, operated_only=None)
    assert image_2d_list[0] + image_2d_list[1] == pytest.approx(
        image_2d_not_operated + 3.0 * image_2d_operated, 1.0e-4
    )


def test__galaxy_image_2d_dict_from(sub_grid_2d_7x7):

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))
    g1 = ag.Galaxy(
        redshift=0.5,
        mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0),
        light_profile=ag.lp.EllSersic(intensity=2.0),
    )

    g2 = ag.Galaxy(
        redshift=0.5, light_profile=ag.lp_operated.EllGaussian(intensity=3.0)
    )

    g0_image = g0.image_2d_from(grid=sub_grid_2d_7x7)
    g1_image = g1.image_2d_from(grid=sub_grid_2d_7x7)
    g2_image = g2.image_2d_from(grid=sub_grid_2d_7x7)

    plane = ag.Plane(redshift=-0.75, galaxies=[g1, g0, g2])

    galaxy_image_2d_dict = plane.galaxy_image_2d_dict_from(grid=sub_grid_2d_7x7)

    assert (galaxy_image_2d_dict[g0] == g0_image).all()
    assert (galaxy_image_2d_dict[g1] == g1_image).all()
    assert (galaxy_image_2d_dict[g2] == g2_image).all()

    galaxy_image_2d_dict = plane.galaxy_image_2d_dict_from(
        grid=sub_grid_2d_7x7, operated_only=True
    )

    assert (galaxy_image_2d_dict[g0] == np.zeros(shape=(36,))).all()
    assert (galaxy_image_2d_dict[g1] == np.zeros(shape=(36,))).all()
    assert (galaxy_image_2d_dict[g2] == g2_image).all()

    galaxy_image_2d_dict = plane.galaxy_image_2d_dict_from(
        grid=sub_grid_2d_7x7, operated_only=False
    )

    assert (galaxy_image_2d_dict[g0] == g0_image).all()
    assert (galaxy_image_2d_dict[g1] == g1_image).all()
    assert (galaxy_image_2d_dict[g2] == np.zeros(shape=(36,))).all()


def test__light_profile_snr__signal_to_noise_via_simulator_correct():

    background_sky_level = 10.0
    exposure_time = 300.0

    grid = ag.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    sersic_0 = ag.lp_snr.EllSersic(
        signal_to_noise_ratio=10.0, centre=(1.0, 1.0), effective_radius=0.01
    )
    sersic_1 = ag.lp_snr.EllSersic(
        signal_to_noise_ratio=20.0, centre=(-1.0, -1.0), effective_radius=0.01
    )

    plane = ag.Plane(
        galaxies=[ag.Galaxy(redshift=0.5, light_0=sersic_0, light_1=sersic_1)]
    )

    simulator = ag.SimulatorImaging(
        exposure_time=exposure_time,
        noise_seed=1,
        background_sky_level=background_sky_level,
    )

    imaging = simulator.via_plane_from(plane=plane, grid=grid)

    assert 9.0 < imaging.signal_to_noise_map.native[0, 2] < 11.0
    assert 11.0 < imaging.signal_to_noise_map.native[2, 0] < 21.0


### Mass Profile Quantities ###


def test__convergence_2d_from(sub_grid_2d_7x7):

    g0 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0))
    g1 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.SphIsothermal(einstein_radius=2.0))

    g0_convergence = g0.convergence_2d_from(grid=sub_grid_2d_7x7)

    g1_convergence = g1.convergence_2d_from(grid=sub_grid_2d_7x7)

    plane = ag.Plane(galaxies=[g0, g1], redshift=None)

    convergence = plane.convergence_2d_from(grid=sub_grid_2d_7x7)

    assert convergence == pytest.approx(g0_convergence + g1_convergence, 1.0e-8)

    # No galaxies

    plane = ag.Plane(galaxies=[], redshift=0.5)

    convergence = plane.convergence_2d_from(grid=sub_grid_2d_7x7)

    assert convergence.sub_shape_slim == sub_grid_2d_7x7.sub_shape_slim

    convergence = plane.convergence_2d_from(grid=sub_grid_2d_7x7)

    assert convergence.sub_shape_native == (14, 14)

    convergence = plane.convergence_2d_from(grid=sub_grid_2d_7x7)

    assert convergence.shape_native == (7, 7)


def test__potential_2d_from(sub_grid_2d_7x7):

    g0 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0))
    g1 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.SphIsothermal(einstein_radius=2.0))

    g0_potential = g0.potential_2d_from(grid=sub_grid_2d_7x7)

    g1_potential = g1.potential_2d_from(grid=sub_grid_2d_7x7)

    plane = ag.Plane(galaxies=[g0, g1], redshift=None)

    potential = plane.potential_2d_from(grid=sub_grid_2d_7x7)

    assert potential == pytest.approx(g0_potential + g1_potential, 1.0e-8)

    # No galaxies

    plane = ag.Plane(galaxies=[], redshift=0.5)

    potential = plane.potential_2d_from(grid=sub_grid_2d_7x7)

    assert potential.sub_shape_slim == sub_grid_2d_7x7.sub_shape_slim

    potential = plane.potential_2d_from(grid=sub_grid_2d_7x7)

    assert potential.sub_shape_native == (14, 14)

    potential = plane.potential_2d_from(grid=sub_grid_2d_7x7)

    assert potential.shape_native == (7, 7)


def test__deflections_yx_2d_from(sub_grid_2d_7x7):
    # Overwrite one value so intensity in each pixel is different
    sub_grid_2d_7x7[5] = np.array([2.0, 2.0])

    g0 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0))
    g1 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.SphIsothermal(einstein_radius=2.0))

    g0_deflections = g0.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

    g1_deflections = g1.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

    plane = ag.Plane(galaxies=[g0, g1], redshift=None)

    deflections = plane.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

    assert deflections == pytest.approx(g0_deflections + g1_deflections, 1.0e-4)

    # No Galaxies

    plane = ag.Plane(redshift=0.5, galaxies=[])

    deflections = plane.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

    assert deflections.shape_native == (7, 7)
    assert (deflections.binned[0, 0] == 0.0).all()
    assert (deflections.binned[0, 1] == 0.0).all()
    assert (deflections.binned[1, 0] == 0.0).all()
    assert (deflections.binned[0] == 0.0).all()


### Hyper Quantities ###


def test__contribution_map_list():

    hyper_model_image = ag.Array2D.manual_native([[2.0, 4.0, 10.0]], pixel_scales=1.0)
    hyper_galaxy_image = ag.Array2D.manual_native([[1.0, 5.0, 8.0]], pixel_scales=1.0)

    hyper_galaxy_0 = ag.HyperGalaxy(contribution_factor=5.0)
    hyper_galaxy_1 = ag.HyperGalaxy(contribution_factor=10.0)

    contribution_map_0 = hyper_galaxy_0.contribution_map_from(
        hyper_model_image=hyper_model_image, hyper_galaxy_image=hyper_galaxy_image
    )

    contribution_map_1 = hyper_galaxy_1.contribution_map_from(
        hyper_model_image=hyper_model_image, hyper_galaxy_image=hyper_galaxy_image
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

    assert (plane.contribution_map_list[0].slim == contribution_map_0).all()

    plane = ag.Plane(redshift=0.5, galaxies=[galaxy_1])

    assert (plane.contribution_map_list[0].slim == contribution_map_1).all()

    plane = ag.Plane(redshift=0.5, galaxies=[galaxy_1, galaxy_0])

    assert (plane.contribution_map_list[0].slim == contribution_map_1).all()
    assert (plane.contribution_map_list[1].slim == contribution_map_0).all()

    assert plane.contribution_map_list[0].slim == pytest.approx(
        [0.20833333, 0.89285714, 1.0], 1.0e-2
    )
    assert plane.contribution_map_list[1].slim == pytest.approx(
        [0.25714286, 1.0, 0.96], 1.0e-2
    )

    assert (sum(plane.contribution_map_list) == plane.contribution_map).all()

    hyper_model_image = ag.Array2D.manual_native([[2.0, 4.0, 10.0]], pixel_scales=1.0)
    hyper_galaxy_image = ag.Array2D.manual_native([[1.0, 5.0, 8.0]], pixel_scales=1.0)

    hyper_galaxy = ag.HyperGalaxy(contribution_factor=5.0)

    contribution_map = hyper_galaxy.contribution_map_from(
        hyper_model_image=hyper_model_image, hyper_galaxy_image=hyper_galaxy_image
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

    assert (plane.contribution_map_list[0].slim == contribution_map).all()
    assert plane.contribution_map_list[1] == None
    assert plane.contribution_map_list[2] == None

    galaxy_1 = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0, galaxy_1])

    assert (galaxy_0.contribution_map == plane.contribution_map).all()

    galaxy_0 = ag.Galaxy(redshift=0.5)

    plane = ag.Plane(redshift=0.5, galaxies=[galaxy_0, galaxy_1])

    assert plane.contribution_map == None


def test__hyper_noise_map_list_from():
    noise_map = ag.Array2D.manual_native(array=[[1.0, 2.0, 3.0]], pixel_scales=1.0)

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

    hyper_noise_map_list = plane.hyper_noise_map_list_from(noise_map=noise_map)

    assert (hyper_noise_map_list[0].slim == np.array([0.0, 2.0, 3.0])).all()
    assert hyper_noise_map_list[1].slim == pytest.approx(
        np.array([0.73468, (2.0 * 0.75) ** 2.0, 3.0 ** 2.0]), 1.0e-4
    )

    noise_map = ag.Array2D.manual_native(array=[[5.0, 3.0, 1.0]], pixel_scales=1.0)

    hyper_model_image = ag.Array2D.manual_native(
        array=[[2.0, 4.0, 10.0]], pixel_scales=1.0
    )
    hyper_galaxy_image = ag.Array2D.manual_native(
        array=[[1.0, 5.0, 8.0]], pixel_scales=1.0
    )

    hyper_galaxy_0 = ag.HyperGalaxy(contribution_factor=5.0)
    hyper_galaxy_1 = ag.HyperGalaxy(contribution_factor=10.0)

    contribution_map_0 = hyper_galaxy_0.contribution_map_from(
        hyper_model_image=hyper_model_image, hyper_galaxy_image=hyper_galaxy_image
    )

    contribution_map_1 = hyper_galaxy_1.contribution_map_from(
        hyper_model_image=hyper_model_image, hyper_galaxy_image=hyper_galaxy_image
    )

    hyper_noise_map_0 = hyper_galaxy_0.hyper_noise_map_from(
        noise_map=noise_map, contribution_map=contribution_map_0
    )

    hyper_noise_map_1 = hyper_galaxy_1.hyper_noise_map_from(
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

    hyper_noise_map_list = plane.hyper_noise_map_list_from(noise_map=noise_map)
    assert (hyper_noise_map_list[0].slim == hyper_noise_map_0).all()
    assert (hyper_noise_map_list[1].slim == np.zeros(shape=(3, 1))).all()

    plane = ag.Plane(redshift=0.5, galaxies=[ag.Galaxy(redshift=0.5), galaxy_1])

    hyper_noise_map_list = plane.hyper_noise_map_list_from(noise_map=noise_map)
    assert (hyper_noise_map_list[0].slim == np.zeros(shape=(3, 1))).all()
    assert (hyper_noise_map_list[1].slim == hyper_noise_map_1).all()

    plane = ag.Plane(
        redshift=0.5,
        galaxies=[ag.Galaxy(redshift=0.5), galaxy_1, galaxy_0, ag.Galaxy(redshift=0.5)],
    )

    hyper_noise_map_list = plane.hyper_noise_map_list_from(noise_map=noise_map)
    assert (hyper_noise_map_list[0].slim == np.zeros(shape=(3, 1))).all()
    assert (hyper_noise_map_list[1].slim == hyper_noise_map_1).all()
    assert (hyper_noise_map_list[2].slim == hyper_noise_map_0).all()
    assert (hyper_noise_map_list[3].slim == np.zeros(shape=(3, 1))).all()

    # Filters Nones from list when computing noise map.

    plane = ag.Plane(
        redshift=0.5,
        galaxies=[ag.Galaxy(redshift=0.5), galaxy_1, galaxy_0, ag.Galaxy(redshift=0.5)],
    )

    hyper_noise_map = plane.hyper_noise_map_from(noise_map=noise_map)
    assert (hyper_noise_map.slim == hyper_noise_map_0 + hyper_noise_map_1).all()

    # No Galaxies

    noise_map = ag.Array2D.manual_native(array=[[5.0, 3.0, 1.0]], pixel_scales=1.0)

    plane = ag.Plane(redshift=0.5, galaxies=[ag.Galaxy(redshift=0.5)])
    hyper_noise_map = plane.hyper_noise_map_from(noise_map=noise_map)

    assert (hyper_noise_map == np.zeros((3, 1))).all()


def test__plane_image_2d_from(sub_grid_2d_7x7):
    sub_grid_2d_7x7[1] = np.array([2.0, 2.0])

    galaxy = ag.Galaxy(redshift=0.5, light=ag.lp.EllSersic(intensity=1.0))

    plane = ag.Plane(galaxies=[galaxy], redshift=None)

    plane_image_from_func = ag.plane.plane.plane_util.plane_image_of_galaxies_from(
        shape=(7, 7), grid=sub_grid_2d_7x7.mask.unmasked_grid_sub_1, galaxies=[galaxy]
    )

    plane_image_from_plane = plane.plane_image_2d_from(grid=sub_grid_2d_7x7)

    assert (plane_image_from_func.array == plane_image_from_plane.array).all()

    # The grid coordinates -2.0 -> 2.0 mean a plane of shape (5,5) has arc second coordinates running over
    # -1.6, -0.8, 0.0, 0.8, 1.6. The origin -1.6, -1.6 of the model_galaxy means its brighest pixel should be
    # index 0 of the 1D grid and (0,0) of the 2d plane data.

    mask = ag.Mask2D.unmasked(shape_native=(5, 5), pixel_scales=1.0, sub_size=1)

    grid = ag.Grid2D.from_mask(mask=mask)

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.lp.EllSersic(centre=(1.6, -1.6), intensity=1.0)
    )
    plane = ag.Plane(galaxies=[g0], redshift=None)

    plane_image = plane.plane_image_2d_from(grid=grid)

    assert plane_image.array.shape_native == (5, 5)
    assert np.unravel_index(
        plane_image.array.native.argmax(), plane_image.array.native.shape
    ) == (0, 0)

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.lp.EllSersic(centre=(1.6, 1.6), intensity=1.0)
    )
    plane = ag.Plane(galaxies=[g0], redshift=None)

    plane_image = plane.plane_image_2d_from(grid=grid)

    assert np.unravel_index(
        plane_image.array.native.argmax(), plane_image.array.native.shape
    ) == (0, 4)

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.lp.EllSersic(centre=(-1.6, -1.6), intensity=1.0)
    )
    plane = ag.Plane(galaxies=[g0], redshift=None)

    plane_image = plane.plane_image_2d_from(grid=grid)

    assert np.unravel_index(
        plane_image.array.native.argmax(), plane_image.array.native.shape
    ) == (4, 0)

    g0 = ag.Galaxy(
        redshift=0.5, light_profile=ag.lp.EllSersic(centre=(-1.6, 1.6), intensity=1.0)
    )
    plane = ag.Plane(galaxies=[g0], redshift=None)

    plane_image = plane.plane_image_2d_from(grid=grid)

    assert np.unravel_index(
        plane_image.array.native.argmax(), plane_image.array.native.shape
    ) == (4, 4)


### Traced Grids ###


def test__traced_grid_from(sub_grid_2d_7x7, sub_grid_2d_7x7_simple, gal_x1_mp):
    # Overwrite one value so intensity in each pixel is different
    sub_grid_2d_7x7[5] = np.array([2.0, 2.0])

    g0 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.SphIsothermal(einstein_radius=1.0))
    g1 = ag.Galaxy(redshift=0.5, mass_profile=ag.mp.SphIsothermal(einstein_radius=2.0))

    g0_deflections = g0.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

    g1_deflections = g1.deflections_yx_2d_from(grid=sub_grid_2d_7x7)

    traced_grid = sub_grid_2d_7x7 - (g0_deflections + g1_deflections)

    plane = ag.Plane(galaxies=[g0, g1], redshift=None)

    plane_traced_grid = plane.traced_grid_from(grid=sub_grid_2d_7x7)

    assert plane_traced_grid == pytest.approx(traced_grid, 1.0e-4)

    plane = ag.Plane(galaxies=[gal_x1_mp, gal_x1_mp], redshift=None)

    traced_grid = plane.traced_grid_from(grid=sub_grid_2d_7x7_simple)

    assert traced_grid[0] == pytest.approx(
        np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3
    )
    assert traced_grid[1] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)
    assert traced_grid[2] == pytest.approx(
        np.array([1.0 - 2.0 * 0.707, 1.0 - 2.0 * 0.707]), 1e-3
    )
    assert traced_grid[3] == pytest.approx(np.array([-1.0, 0.0]), 1e-3)

    # No Galaxies

    plane = ag.Plane(galaxies=[], redshift=1.0)

    traced_grid = plane.traced_grid_from(grid=sub_grid_2d_7x7)

    assert (traced_grid == sub_grid_2d_7x7).all()


### Extract ###


def test__extract_attribute():

    g0 = ag.Galaxy(
        redshift=0.5, mp_0=ag.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
    )
    g1 = ag.Galaxy(
        redshift=0.5, mp_0=ag.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
    )
    g2 = ag.Galaxy(
        redshift=0.5,
        mp_0=ag.m.MockMassProfile(value=0.7),
        mp_1=ag.m.MockMassProfile(value=0.6),
    )

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

    values = plane.extract_attribute(cls=ag.mp.MassProfile, attr_name="value")

    assert values == None

    plane = ag.Plane(galaxies=[g0, g1], redshift=None)

    values = plane.extract_attribute(cls=ag.mp.MassProfile, attr_name="value1")

    assert values.in_list == [(1.0, 1.0), (2.0, 2.0)]

    plane = ag.Plane(
        galaxies=[g0, ag.Galaxy(redshift=0.5), g1, ag.Galaxy(redshift=0.5), g2],
        redshift=None,
    )

    values = plane.extract_attribute(cls=ag.mp.MassProfile, attr_name="value")

    assert values.in_list == [0.9, 0.8, 0.7, 0.6]

    plane.extract_attribute(cls=ag.mp.MassProfile, attr_name="incorrect_value")


def test__extract_attributes_of_galaxies():

    g0 = ag.Galaxy(
        redshift=0.5, mp_0=ag.m.MockMassProfile(value=0.9, value1=(1.0, 1.0))
    )
    g1 = ag.Galaxy(
        redshift=0.5, mp_0=ag.m.MockMassProfile(value=0.8, value1=(2.0, 2.0))
    )
    g2 = ag.Galaxy(
        redshift=0.5,
        mp_0=ag.m.MockMassProfile(value=0.7),
        mp_1=ag.m.MockMassProfile(value=0.6),
    )

    plane = ag.Plane(galaxies=[ag.Galaxy(redshift=0.5)], redshift=None)

    values = plane.extract_attributes_of_galaxies(
        cls=ag.mp.MassProfile, attr_name="value"
    )

    assert values == [None]

    plane = ag.Plane(galaxies=[g0, g1], redshift=None)

    values = plane.extract_attributes_of_galaxies(
        cls=ag.mp.MassProfile, attr_name="value1"
    )

    assert values[0].in_list == [(1.0, 1.0)]
    assert values[1].in_list == [(2.0, 2.0)]

    plane = ag.Plane(
        galaxies=[g0, ag.Galaxy(redshift=0.5), g1, ag.Galaxy(redshift=0.5), g2],
        redshift=None,
    )

    values = plane.extract_attributes_of_galaxies(
        cls=ag.mp.MassProfile, attr_name="value", filter_nones=False
    )

    assert values[0].in_list == [0.9]
    assert values[1] == None
    assert values[2].in_list == [0.8]
    assert values[3] == None
    assert values[4].in_list == [0.7, 0.6]

    values = plane.extract_attributes_of_galaxies(
        cls=ag.mp.MassProfile, attr_name="value", filter_nones=True
    )

    assert values[0].in_list == [0.9]
    assert values[1].in_list == [0.8]
    assert values[2].in_list == [0.7, 0.6]

    plane.extract_attributes_of_galaxies(
        cls=ag.mp.MassProfile, attr_name="incorrect_value", filter_nones=True
    )


### Exceptions ###


def test__no_galaxies__raises_exception_if_no_plane_redshift_input():
    plane = ag.Plane(galaxies=[], redshift=0.5)
    assert plane.redshift == 0.5

    with pytest.raises(exc.PlaneException):
        ag.Plane(galaxies=[])


def test__galaxy_redshifts_gives_list_of_redshifts():
    g0 = ag.Galaxy(redshift=1.0)
    g1 = ag.Galaxy(redshift=1.0)
    g2 = ag.Galaxy(redshift=1.0)

    plane = ag.Plane(galaxies=[g0, g1, g2])

    assert plane.redshift == 1.0
    assert plane.galaxy_redshifts == [1.0, 1.0, 1.0]


### Decorators ###


def test__grid_iterate_in__iterates_grid_correctly(gal_x1_lp):

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

    grid = ag.Grid2DIterate.from_mask(mask=mask, fractional_accuracy=1.0, sub_steps=[2])

    plane = ag.Plane(galaxies=[gal_x1_lp], redshift=None)

    image = plane.image_2d_from(grid=grid)

    mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
    grid_sub_2 = ag.Grid2D.from_mask(mask=mask_sub_2)
    image_sub_2 = plane.image_2d_from(grid=grid_sub_2).binned

    assert (image == image_sub_2).all()

    grid = ag.Grid2DIterate.from_mask(
        mask=mask, fractional_accuracy=0.95, sub_steps=[2, 4, 8]
    )

    galaxy = ag.Galaxy(
        redshift=0.5, light=ag.lp.EllSersic(centre=(0.08, 0.08), intensity=1.0)
    )

    plane = ag.Plane(galaxies=[galaxy])

    image = plane.image_2d_from(grid=grid)

    mask_sub_4 = mask.mask_new_sub_size_from(mask=mask, sub_size=4)
    grid_sub_4 = ag.Grid2D.from_mask(mask=mask_sub_4)
    image_sub_4 = plane.image_2d_from(grid=grid_sub_4).binned

    assert image[0] == image_sub_4[0]

    mask_sub_8 = mask.mask_new_sub_size_from(mask=mask, sub_size=8)
    grid_sub_8 = ag.Grid2D.from_mask(mask=mask_sub_8)
    image_sub_8 = plane.image_2d_from(grid=grid_sub_8).binned

    assert image[4] == image_sub_8[4]


def test__grid_iterate_in__iterates_grid_result_correctly(gal_x1_mp):

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

    grid = ag.Grid2DIterate.from_mask(mask=mask, fractional_accuracy=1.0, sub_steps=[2])

    galaxy = ag.Galaxy(
        redshift=0.5, mass=ag.mp.EllIsothermal(centre=(0.08, 0.08), einstein_radius=1.0)
    )

    plane = ag.Plane(galaxies=[galaxy], redshift=None)

    deflections = plane.deflections_yx_2d_from(grid=grid)

    mask_sub_2 = mask.mask_new_sub_size_from(mask=mask, sub_size=2)
    grid_sub_2 = ag.Grid2D.from_mask(mask=mask_sub_2)
    deflections_sub_2 = galaxy.deflections_yx_2d_from(grid=grid_sub_2).binned

    assert (deflections == deflections_sub_2).all()

    grid = ag.Grid2DIterate.from_mask(
        mask=mask, fractional_accuracy=0.99, sub_steps=[2, 4, 8]
    )

    galaxy = ag.Galaxy(
        redshift=0.5, mass=ag.mp.EllIsothermal(centre=(0.08, 0.08), einstein_radius=1.0)
    )

    plane = ag.Plane(galaxies=[galaxy], redshift=None)

    deflections = plane.deflections_yx_2d_from(grid=grid)

    mask_sub_4 = mask.mask_new_sub_size_from(mask=mask, sub_size=4)
    grid_sub_4 = ag.Grid2D.from_mask(mask=mask_sub_4)
    deflections_sub_4 = galaxy.deflections_yx_2d_from(grid=grid_sub_4).binned

    assert deflections[0, 0] == deflections_sub_4[0, 0]

    mask_sub_8 = mask.mask_new_sub_size_from(mask=mask, sub_size=8)
    grid_sub_8 = ag.Grid2D.from_mask(mask=mask_sub_8)
    deflections_sub_8 = galaxy.deflections_yx_2d_from(grid=grid_sub_8).binned

    assert deflections[4, 0] == deflections_sub_8[4, 0]


### Regression ###


def test__centre_of_profile_in_right_place():
    grid = ag.Grid2D.uniform(shape_native=(7, 7), pixel_scales=1.0)

    galaxy = ag.Galaxy(
        redshift=0.5,
        mass=ag.mp.EllIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        mass_0=ag.mp.EllIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
    )

    plane = ag.Plane(galaxies=[galaxy])

    convergence = plane.convergence_2d_from(grid=grid)
    max_indexes = np.unravel_index(
        convergence.native.argmax(), convergence.shape_native
    )
    assert max_indexes == (1, 4)

    potential = plane.potential_2d_from(grid=grid)
    max_indexes = np.unravel_index(potential.native.argmin(), potential.shape_native)
    assert max_indexes == (1, 4)

    deflections = plane.deflections_yx_2d_from(grid=grid)
    assert deflections.native[1, 4, 0] > 0
    assert deflections.native[2, 4, 0] < 0
    assert deflections.native[1, 4, 1] > 0
    assert deflections.native[1, 3, 1] < 0

    galaxy = ag.Galaxy(
        redshift=0.5,
        mass=ag.mp.SphIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        mass_0=ag.mp.SphIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
    )

    plane = ag.Plane(galaxies=[galaxy])

    convergence = plane.convergence_2d_from(grid=grid)
    max_indexes = np.unravel_index(
        convergence.native.argmax(), convergence.shape_native
    )
    assert max_indexes == (1, 4)

    potential = plane.potential_2d_from(grid=grid)
    max_indexes = np.unravel_index(potential.native.argmin(), potential.shape_native)
    assert max_indexes == (1, 4)

    deflections = plane.deflections_yx_2d_from(grid=grid)
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
        mass=ag.mp.EllIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
        mass_0=ag.mp.EllIsothermal(centre=(2.0, 1.0), einstein_radius=1.0),
    )

    plane = ag.Plane(galaxies=[galaxy])

    convergence = plane.convergence_2d_from(grid=grid)
    max_indexes = np.unravel_index(
        convergence.native.argmax(), convergence.shape_native
    )
    assert max_indexes == (1, 4)

    potential = plane.potential_2d_from(grid=grid)
    max_indexes = np.unravel_index(potential.native.argmin(), potential.shape_native)
    assert max_indexes == (1, 4)

    deflections = plane.deflections_yx_2d_from(grid=grid)
    assert deflections.native[1, 4, 0] >= 0
    assert deflections.native[2, 4, 0] <= 0
    assert deflections.native[1, 4, 1] >= 0
    assert deflections.native[1, 3, 1] <= 0

    galaxy = ag.Galaxy(
        redshift=0.5, mass=ag.mp.SphIsothermal(centre=(2.0, 1.0), einstein_radius=1.0)
    )

    plane = ag.Plane(galaxies=[galaxy])

    convergence = plane.convergence_2d_from(grid=grid)
    max_indexes = np.unravel_index(
        convergence.native.argmax(), convergence.shape_native
    )
    assert max_indexes == (1, 4)

    potential = plane.potential_2d_from(grid=grid)
    max_indexes = np.unravel_index(potential.native.argmin(), potential.shape_native)
    assert max_indexes == (1, 4)

    deflections = plane.deflections_yx_2d_from(grid=grid)
    assert deflections.native[1, 4, 0] >= 0
    assert deflections.native[2, 4, 0] <= 0
    assert deflections.native[1, 4, 1] >= 0
    assert deflections.native[1, 3, 1] <= 0
