import numpy as np

import autoarray as aa

from autogalaxy import exc


class Clicker:
    def __init__(self, image, pixel_scales, search_box_size):

        self.image = image

        pixel_scales = aa.util.geometry.convert_pixel_scales_2d(
            pixel_scales=pixel_scales
        )

        self.pixel_scales = pixel_scales
        self.search_box_size = search_box_size

        self.click_list = []

    def onclick(self, event):

        if event.dblclick:

            y_arcsec = (
                np.rint(event.ydata / self.pixel_scales[0]) * self.pixel_scales[0]
            )
            x_arcsec = (
                np.rint(event.xdata / self.pixel_scales[1]) * self.pixel_scales[1]
            )

            (y_pixels, x_pixels) = self.image.mask.pixel_coordinates_2d_from(
                scaled_coordinates_2d=(y_arcsec, x_arcsec)
            )

            flux = -np.inf

            for y in range(
                y_pixels - self.search_box_size, y_pixels + self.search_box_size
            ):
                for x in range(
                    x_pixels - self.search_box_size, x_pixels + self.search_box_size
                ):

                    flux_new = self.image.native[y, x]

                    if flux_new > flux:
                        flux = flux_new
                        y_pixels_max = y
                        x_pixels_max = x

            grid_arcsec = self.image.mask.grid_scaled_from(
                grid_pixels_1d=aa.Grid2D.manual_native(
                    grid=[[[y_pixels_max + 0.5, x_pixels_max + 0.5]]],
                    pixel_scales=self.pixel_scales,
                )
            )
            y_arcsec = grid_arcsec[0, 0]
            x_arcsec = grid_arcsec[0, 1]

            print("clicked on:", y_pixels, x_pixels)
            print("Max flux pixel:", y_pixels_max, x_pixels_max)
            print("Arc-sec Coordinate", y_arcsec, x_arcsec)

            self.click_list.append((y_arcsec, x_arcsec))
