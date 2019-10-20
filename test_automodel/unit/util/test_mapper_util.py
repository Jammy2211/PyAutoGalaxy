import automodel as am
import numpy as np
import pytest


@pytest.fixture(name="three_pixels")
def make_three_pixels():
    return np.array([[0, 0], [0, 1], [1, 0]])


@pytest.fixture(name="five_pixels")
def make_five_pixels():
    return np.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])


class TestMappingMatrix:
    def test__3_image_pixels__6_pixel_pixels__sub_grid_1x1(self, three_pixels):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2])
        mask_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2])

        mapping_matrix = am.util.mapper.mapping_matrix_from_pixelization_1d_index_for_sub_mask_1d_index(
            pixelization_1d_index_for_sub_mask_1d_index=pixelization_1d_index_for_sub_mask_1d_index,
            pixels=6,
            total_mask_pixels=3,
            mask_1d_index_for_sub_mask_1d_index=mask_1d_index_for_sub_mask_1d_index,
            sub_fraction=1.0,
        )

        assert (
            mapping_matrix
            == np.array(
                [
                    [1, 0, 0, 0, 0, 0],  # Imaging pixel 0 maps to pix pixel 0.
                    [0, 1, 0, 0, 0, 0],  # Imaging pixel 1 maps to pix pixel 1.
                    [0, 0, 1, 0, 0, 0],
                ]
            )
        ).all()  # Imaging pixel 2 maps to pix pixel 2

    def test__5_image_pixels__8_pixel_pixels__sub_grid_1x1(self, five_pixels):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2, 7, 6])
        mask_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2, 3, 4])

        mapping_matrix = am.util.mapper.mapping_matrix_from_pixelization_1d_index_for_sub_mask_1d_index(
            pixelization_1d_index_for_sub_mask_1d_index=pixelization_1d_index_for_sub_mask_1d_index,
            pixels=8,
            total_mask_pixels=5,
            mask_1d_index_for_sub_mask_1d_index=mask_1d_index_for_sub_mask_1d_index,
            sub_fraction=1.0,
        )

        assert (
            mapping_matrix
            == np.array(
                [
                    [
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],  # Imaging image_to_pixel 0 and 3 mappers to pix pixel 0.
                    [
                        0,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],  # Imaging image_to_pixel 1 and 4 mappers to pix pixel 1.
                    [0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                ]
            )
        ).all()  # Imaging image_to_pixel 2 and 5 mappers to pix pixel 2

    def test__5_image_pixels__8_pixel_pixels__sub_grid_2x2__no_overlapping_pixels(
        self, five_pixels
    ):

        pixelization_1d_index_for_sub_mask_1d_index = np.array(
            [0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2]
        )
        mask_1d_index_for_sub_mask_1d_index = np.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        )
        mapping_matrix = am.util.mapper.mapping_matrix_from_pixelization_1d_index_for_sub_mask_1d_index(
            pixelization_1d_index_for_sub_mask_1d_index=pixelization_1d_index_for_sub_mask_1d_index,
            pixels=8,
            total_mask_pixels=5,
            mask_1d_index_for_sub_mask_1d_index=mask_1d_index_for_sub_mask_1d_index,
            sub_fraction=0.25,
        )

        assert (
            mapping_matrix
            == np.array(
                [
                    [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0],
                    [0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0],
                    [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
                    [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
                    [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25],
                ]
            )
        ).all()

    def test__5_image_pixels__8_pixel_pixels__sub_grid_2x2__include_overlapping_pixels(
        self, five_pixels
    ):

        pixelization_1d_index_for_sub_mask_1d_index = np.array(
            [0, 0, 0, 1, 1, 1, 0, 0, 2, 3, 4, 5, 7, 0, 1, 3, 6, 7, 4, 2]
        )
        mask_1d_index_for_sub_mask_1d_index = np.array(
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
        )

        mapping_matrix = am.util.mapper.mapping_matrix_from_pixelization_1d_index_for_sub_mask_1d_index(
            pixelization_1d_index_for_sub_mask_1d_index=pixelization_1d_index_for_sub_mask_1d_index,
            pixels=8,
            total_mask_pixels=5,
            mask_1d_index_for_sub_mask_1d_index=mask_1d_index_for_sub_mask_1d_index,
            sub_fraction=0.25,
        )

        assert (
            mapping_matrix
            == np.array(
                [
                    [0.75, 0.25, 0, 0, 0, 0, 0, 0],
                    [0.5, 0.5, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0],
                    [0.25, 0.25, 0, 0.25, 0, 0, 0, 0.25],
                    [0, 0, 0.25, 0, 0.25, 0, 0.25, 0.25],
                ]
            )
        ).all()

    def test__3_image_pixels__6_pixel_pixels__sub_grid_4x4(self, three_pixels):

        pixelization_1d_index_for_sub_mask_1d_index = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
                4,
                5,
                0,
                1,
                2,
                3,
            ]
        )

        mask_1d_index_for_sub_mask_1d_index = np.array(
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
                2,
            ]
        )

        mapping_matrix = am.util.mapper.mapping_matrix_from_pixelization_1d_index_for_sub_mask_1d_index(
            pixelization_1d_index_for_sub_mask_1d_index=pixelization_1d_index_for_sub_mask_1d_index,
            pixels=6,
            total_mask_pixels=3,
            mask_1d_index_for_sub_mask_1d_index=mask_1d_index_for_sub_mask_1d_index,
            sub_fraction=1.0 / 16.0,
        )

        assert (
            mapping_matrix
            == np.array(
                [
                    [0.75, 0.25, 0, 0, 0, 0],
                    [0, 0, 1.0, 0, 0, 0],
                    [0.1875, 0.1875, 0.1875, 0.1875, 0.125, 0.125],
                ]
            )
        ).all()


class TestPixelSignals:
    def test__x3_image_pixels_signals_1s__pixel_scale_1__pixel_signals_all_1s(self):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2])
        mask_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2])
        galaxy_image = np.array([1.0, 1.0, 1.0])

        pixel_signals = am.util.mapper.adaptive_pixel_signals_from_images(
            pixels=3,
            signal_scale=1.0,
            pixelization_1d_index_for_sub_mask_1d_index=pixelization_1d_index_for_sub_mask_1d_index,
            mask_1d_index_for_sub_mask_1d_index=mask_1d_index_for_sub_mask_1d_index,
            hyper_image=galaxy_image,
        )

        assert (pixel_signals == np.array([1.0, 1.0, 1.0])).all()

    def test__x4_image_pixels_signals_1s__pixel_signals_still_all_1s(self):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2, 0])
        mask_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2, 0])
        galaxy_image = np.array([1.0, 1.0, 1.0, 1.0])

        pixel_signals = am.util.mapper.adaptive_pixel_signals_from_images(
            pixels=3,
            signal_scale=1.0,
            pixelization_1d_index_for_sub_mask_1d_index=pixelization_1d_index_for_sub_mask_1d_index,
            mask_1d_index_for_sub_mask_1d_index=mask_1d_index_for_sub_mask_1d_index,
            hyper_image=galaxy_image,
        )

        assert (pixel_signals == np.array([1.0, 1.0, 1.0])).all()

    def test__galaxy_flux_in_a_pixel_pixel_is_double_the_others__pixel_signal_is_1_others_a_half(
        self
    ):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2])
        mask_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2])
        galaxy_image = np.array([2.0, 1.0, 1.0])

        pixel_signals = am.util.mapper.adaptive_pixel_signals_from_images(
            pixels=3,
            signal_scale=1.0,
            pixelization_1d_index_for_sub_mask_1d_index=pixelization_1d_index_for_sub_mask_1d_index,
            mask_1d_index_for_sub_mask_1d_index=mask_1d_index_for_sub_mask_1d_index,
            hyper_image=galaxy_image,
        )

        assert (pixel_signals == np.array([1.0, 0.5, 0.5])).all()

    def test__same_as_above_but_pixel_scale_2__scales_pixel_signals(self):

        pixelization_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2])
        mask_1d_index_for_sub_mask_1d_index = np.array([0, 1, 2])
        galaxy_image = np.array([2.0, 1.0, 1.0])

        pixel_signals = am.util.mapper.adaptive_pixel_signals_from_images(
            pixels=3,
            signal_scale=2.0,
            pixelization_1d_index_for_sub_mask_1d_index=pixelization_1d_index_for_sub_mask_1d_index,
            mask_1d_index_for_sub_mask_1d_index=mask_1d_index_for_sub_mask_1d_index,
            hyper_image=galaxy_image,
        )

        assert (pixel_signals == np.array([1.0, 0.25, 0.25])).all()