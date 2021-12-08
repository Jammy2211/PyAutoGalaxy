from __future__ import division, print_function

import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__blurred_image_2d_via_psf_from(
    sub_grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3, convolver_7x7
):
    lp = ag.lp.EllSersic(intensity=1.0)

    operate_lp = ag.OperateImage(light_obj_list=[lp])

    lp_blurred_image_2d = operate_lp.blurred_image_2d_via_psf_from(
        grid=sub_grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    image_2d = lp.image_2d_from(grid=sub_grid_2d_7x7)
    blurring_image_2d = lp.image_2d_from(grid=blurring_grid_2d_7x7)
    blurred_image_2d = convolver_7x7.convolve_image(
        image=image_2d.binned, blurring_image=blurring_image_2d.binned
    )

    assert blurred_image_2d.slim == pytest.approx(lp_blurred_image_2d.slim, 1.0e-4)
    assert blurred_image_2d.native == pytest.approx(lp_blurred_image_2d.native, 1.0e-4)


def test__blurred_image_2d_list_via_psf_from(
    sub_grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3
):
    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=2.0))

    g0_calc_image = ag.OperateImage(light_obj_list=[g0])
    g1_calc_image = ag.OperateImage(light_obj_list=[g1])

    blurred_g0_image = g0_calc_image.blurred_image_2d_via_psf_from(
        grid=sub_grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    blurred_g1_image = g1_calc_image.blurred_image_2d_via_psf_from(
        grid=sub_grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    operate_list = ag.OperateImage(light_obj_list=[g0, g1])

    blurred_image_list = operate_list.blurred_image_2d_list_via_psf_from(
        grid=sub_grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    assert blurred_g0_image.shape_slim == 9
    assert blurred_image_list[0].slim == pytest.approx(blurred_g0_image.slim, 1.0e-4)
    assert blurred_g1_image.shape_slim == 9
    assert blurred_image_list[1].slim == pytest.approx(blurred_g1_image.slim, 1.0e-4)

    assert blurred_image_list[0].native == pytest.approx(
        blurred_g0_image.native, 1.0e-4
    )
    assert blurred_image_list[1].native == pytest.approx(
        blurred_g1_image.native, 1.0e-4
    )


def test__blurred_image_2d_via_convolver_from(
    sub_grid_2d_7x7, blurring_grid_2d_7x7, convolver_7x7
):
    lp = ag.lp.EllSersic(intensity=1.0)
    operate_lp = ag.OperateImage(light_obj_list=[lp])

    lp_blurred_image_2d = operate_lp.blurred_image_2d_via_convolver_from(
        grid=sub_grid_2d_7x7,
        convolver=convolver_7x7,
        blurring_grid=blurring_grid_2d_7x7,
    )

    image_2d = lp.image_2d_from(grid=sub_grid_2d_7x7)
    blurring_image_2d = lp.image_2d_from(grid=blurring_grid_2d_7x7)
    blurred_image_2d = convolver_7x7.convolve_image(
        image=image_2d.binned, blurring_image=blurring_image_2d.binned
    )

    assert blurred_image_2d.slim == pytest.approx(lp_blurred_image_2d.slim, 1.0e-4)
    assert blurred_image_2d.native == pytest.approx(lp_blurred_image_2d.native, 1.0e-4)


def test__blurred_image_2d_list_via_convolver_from(
    sub_grid_2d_7x7, blurring_grid_2d_7x7, convolver_7x7
):
    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=2.0))

    g0_calc_image = ag.OperateImage(light_obj_list=[g0])
    g1_calc_image = ag.OperateImage(light_obj_list=[g1])

    blurred_g0_image = g0_calc_image.blurred_image_2d_via_convolver_from(
        grid=sub_grid_2d_7x7,
        convolver=convolver_7x7,
        blurring_grid=blurring_grid_2d_7x7,
    )

    blurred_g1_image = g1_calc_image.blurred_image_2d_via_convolver_from(
        grid=sub_grid_2d_7x7,
        convolver=convolver_7x7,
        blurring_grid=blurring_grid_2d_7x7,
    )

    operate_list = ag.OperateImage(light_obj_list=[g0, g1])

    blurred_images_of_galaxies = operate_list.blurred_image_2d_list_via_convolver_from(
        grid=sub_grid_2d_7x7,
        blurring_grid=blurring_grid_2d_7x7,
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


def test__unmasked_blurred_image_2d_via_psf_from():

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

    lp = ag.lp.EllSersic(intensity=0.1)
    operate_lp = ag.OperateImage(light_obj_list=[lp])

    unmasked_blurred_image_2d = operate_lp.unmasked_blurred_image_2d_via_psf_from(
        grid=grid, psf=psf
    )

    assert unmasked_blurred_image_2d.native == pytest.approx(
        np.array(
            [
                [1.31618566e-01, 2.48460648e02, 1.91885830e-01],
                [9.66709359e-02, 8.29737297e01, 1.65678804e02],
                [4.12176698e-02, 8.74484826e-02, 1.01484934e-01],
            ]
        ),
        1.0e-4,
    )


def test__unmasked_blurred_image_2d_list_via_psf_from():
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

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=0.1))
    g1 = ag.Galaxy(redshift=1.0, light_profile=ag.lp.EllSersic(intensity=0.2))

    plane = ag.Plane(redshift=0.75, galaxies=[g0, g1])

    padded_grid = grid.padded_grid_from(kernel_shape_native=psf.shape_native)

    manual_blurred_image_0 = plane.images_of_galaxies_from(grid=padded_grid)[0]
    manual_blurred_image_0 = psf.convolved_array_from(array=manual_blurred_image_0)

    manual_blurred_image_1 = plane.images_of_galaxies_from(grid=padded_grid)[1]
    manual_blurred_image_1 = psf.convolved_array_from(array=manual_blurred_image_1)

    operate_list = ag.OperateImage(light_obj_list=[g0, g1])

    unmasked_blurred_image_2d_list = operate_list.unmasked_blurred_image_2d_list_via_psf_from(
        grid=grid, psf=psf
    )

    assert unmasked_blurred_image_2d_list[0].native == pytest.approx(
        manual_blurred_image_0.binned.native[1:4, 1:4], 1.0e-4
    )

    assert unmasked_blurred_image_2d_list[1].native == pytest.approx(
        manual_blurred_image_1.binned.native[1:4, 1:4], 1.0e-4
    )


def test__visibilities_from_grid_and_transformer(
    grid_2d_7x7, sub_grid_2d_7x7, transformer_7x7_7
):
    lp = ag.lp.EllSersic(intensity=1.0)
    operate_lp = ag.OperateImage(light_obj_list=[lp])

    lp_visibilities = operate_lp.visibilities_via_transformer_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    image_2d = lp.image_2d_from(grid=grid_2d_7x7)
    visibilities = transformer_7x7_7.visibilities_from(image=image_2d.binned)

    assert visibilities == pytest.approx(lp_visibilities, 1.0e-4)


def test__visibilities_list_via_transformer_from(sub_grid_2d_7x7, transformer_7x7_7):

    g0 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=1.0))
    g1 = ag.Galaxy(redshift=0.5, light_profile=ag.lp.EllSersic(intensity=2.0))

    g0_image = g0.image_2d_from(grid=sub_grid_2d_7x7)
    g1_image = g1.image_2d_from(grid=sub_grid_2d_7x7)

    g0_visibilities = transformer_7x7_7.visibilities_from(image=g0_image)
    g1_visibilities = transformer_7x7_7.visibilities_from(image=g1_image)

    operate_list = ag.OperateImage(light_obj_list=[g0, g1])

    visibilities_list = operate_list.visibilities_list_via_transformer_from(
        grid=sub_grid_2d_7x7, transformer=transformer_7x7_7
    )

    assert (g0_visibilities == visibilities_list[0]).all()
    assert (g1_visibilities == visibilities_list[1]).all()
