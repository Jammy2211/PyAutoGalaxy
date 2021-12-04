from __future__ import division, print_function

import numpy as np
import pytest

import autogalaxy as ag

grid = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__blurred_image_2d_via_psf_from(
    sub_grid_2d_7x7, blurring_grid_2d_7x7, psf_3x3, convolver_7x7
):
    lp = ag.lp.EllSersic(intensity=1.0)

    image = lp.image_2d_from(grid=sub_grid_2d_7x7)

    blurring_image = lp.image_2d_from(grid=blurring_grid_2d_7x7)

    blurred_image = convolver_7x7.convolve_image(
        image=image.binned, blurring_image=blurring_image.binned
    )

    light_profile_blurred_image = lp.blurred_image_2d_via_psf_from(
        grid=sub_grid_2d_7x7, blurring_grid=blurring_grid_2d_7x7, psf=psf_3x3
    )

    assert blurred_image.slim == pytest.approx(light_profile_blurred_image.slim, 1.0e-4)
    assert blurred_image.native == pytest.approx(
        light_profile_blurred_image.native, 1.0e-4
    )


def test__blurred_image_2d_via_convolver_from(
    sub_grid_2d_7x7, blurring_grid_2d_7x7, convolver_7x7
):
    lp = ag.lp.EllSersic(intensity=1.0)

    image = lp.image_2d_from(grid=sub_grid_2d_7x7)

    blurring_image = lp.image_2d_from(grid=blurring_grid_2d_7x7)

    blurred_image = convolver_7x7.convolve_image(
        image=image.binned, blurring_image=blurring_image.binned
    )

    light_profile_blurred_image = lp.blurred_image_2d_via_convolver_from(
        grid=sub_grid_2d_7x7,
        convolver=convolver_7x7,
        blurring_grid=blurring_grid_2d_7x7,
    )

    assert blurred_image.slim == pytest.approx(light_profile_blurred_image.slim, 1.0e-4)
    assert blurred_image.native == pytest.approx(
        light_profile_blurred_image.native, 1.0e-4
    )


def test__visibilities_from_grid_and_transformer(
    grid_2d_7x7, sub_grid_2d_7x7, transformer_7x7_7
):
    lp = ag.lp.EllSersic(intensity=1.0)

    image = lp.image_2d_from(grid=grid_2d_7x7)

    visibilities = transformer_7x7_7.visibilities_from(image=image.binned)

    light_profile_visibilities = lp.profile_visibilities_via_transformer_from(
        grid=grid_2d_7x7, transformer=transformer_7x7_7
    )

    assert visibilities == pytest.approx(light_profile_visibilities, 1.0e-4)


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

    unmasked_blurred_image_2d = lp.unmasked_blurred_image_2d_via_psf_from(
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
