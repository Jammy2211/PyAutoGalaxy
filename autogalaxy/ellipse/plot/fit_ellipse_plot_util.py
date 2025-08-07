from astropy import units
import itertools
import matplotlib.pyplot as plt
import numpy as np


def plot_ellipse_residuals(array, fit_list, colors, output, for_subplot: bool = False):

    color = itertools.cycle(colors)

    if for_subplot:
        ax = plt.subplot(1, 2, 2)
    else:

        fig = plt.figure(figsize=(8, 8))

        ax = fig.add_subplot(111)

    for i, fit in enumerate(fit_list):

        angles = fit.ellipse.angles_from_x0_from(pixel_scale=array.pixel_scales[0])

        color_plot = next(color)

        plt.errorbar(
            angles * units.rad.to(units.deg),
            fit.data_interp,
            yerr=fit.noise_map_interp,
            linestyle="None",
            marker="o",
            markersize=2.5,
            color=color_plot,
        )
        plt.axhline(
            np.nanmean(fit.data_interp),
            linestyle=":",
            color=color_plot,
        )

    ax.set_xticks([0, 90, 180, 270, 360])

    ax.set_xlabel(
        r"$\phi$ (deg)",
        fontsize=20,
    )
    ax.set_ylabel(
        r"$\rm I(\phi)$ [E/s]",
        fontsize=20,
        labelpad=-5.0,
    )
    ax.tick_params(axis="y", labelsize=12.5)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    ax.minorticks_on()
    ax.tick_params(
        axis="both",
        which="major",
        length=6,
        width=2,
        right=True,
        top=True,
        direction="in",
        colors="black",
        labelsize=15,
    )
    ax.tick_params(
        axis="both",
        which="minor",
        length=3,
        width=1,
        right=True,
        top=True,
        direction="in",
        colors="black",
        labelsize=15,
    )

    ax.set_yscale("log")

    ax.set_title("Ellipse 1D Residuals", fontsize=20)

    if not for_subplot:

        bbox_original = output.bbox_inches
        output.bbox_inches = None

        output.to_figure(structure=None, auto_filename="ellipse_residuals")

        output.bbox_inches = bbox_original

        plt.close()
