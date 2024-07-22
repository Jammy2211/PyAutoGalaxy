import copy, matplotlib
import numpy as np
import matplotlib.pyplot as plt

from astropy import units
from matplotlib.colors import LinearSegmentedColormap


def subplot_fit(fit_list):
    # NOTE:
    logscale = False

    # NOTE:
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    image = fit.data.native
    image_temp = image

    # NOTE:
    # image = copy.copy(sample.image)
    # if sample.mask is not None:
    #     image[sample.mask.astype(bool)] = np.nan
    #
    #     image_temp = copy.copy(sample.image)
    #     image_temp[~sample.mask.astype(bool)] = np.nan
    # else:
    #     image_temp = None

    # NOTE:
    def custom_colormap():
        # Define the colors
        # colors = ['white', 'black', 'red']
        # positions = [0.0, 0.5, 1.0]
        colors = ["grey", "white", "red", "darkred", "black"]
        positions = [0.0, 0.25, 0.5, 0.75, 1.0]

        # Create the colormap
        cmap = LinearSegmentedColormap.from_list(
            "custom_colormap", list(zip(positions, colors))
        )

        return cmap

    vmin = None
    vmax = None

    if vmin is None:
        vmin = 0.025
    if vmax is None:
        vmax = 5.0

    norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax)

    im = axes[0].imshow(
        np.log10(image) if logscale else image,
        cmap="jet",
        aspect="auto",
        norm=norm,
    )
    if image_temp is not None:
        axes[0].imshow(
            np.log10(image_temp) if logscale else image_temp,
            cmap="jet",
            aspect="auto",
            alpha=0.5,
        )

    ellipse_list = [fit.ellipse for fit in fit_list]

    list_of_angles = []
    list_of_y_fit = []
    list_of_y_errors_fit = []
    y_means = []
    y_stds = []

    radii

    for i, (a, parameters) in enumerate(zip(array, list_of_parameters)):
        if i == 0:
            m = sum(1 for value in parameters.values() if value is not None)

        y_fit, y_errors_fit, (x, y), angles = sample.extract(
            a=a, parameters=parameters, condition=extract_condition
        )
        list_of_angles.append(angles)
        list_of_y_fit.append(y_fit)
        list_of_y_errors_fit.append(y_errors_fit)

        if y_errors_fit is None:
            # y_errors_fit = 0.05 * y_fit
            raise NotImplementedError()

        # NOTE:
        y_mean = np.nanmean(y_fit)
        y_means.append(y_mean)

        # NOTE:
        y_std = np.nanstd(y_fit)
        y_stds.append(y_std)

        # NOTE:
        axes[0].plot(x, y, marker="o", markersize=2.5, color="w")

        # NOTE:
        axes[1].errorbar(
            angles * units.rad.to(units.deg),
            y_fit,
            yerr=y_errors_fit,
            linestyle="None",
            marker="o",
            markersize=2.5,
            color="black",
        )
        axes[1].axhline(y_mean, linestyle=":", color="black")

    levels = np.sort(np.log10(y_means)) if logscale else np.sort(y_means)
    axes[0].contour(
        np.log10(image) if logscale else image,
        # levels=y_means[::-1],
        levels=levels,
        colors="black",
    )
    colors = [im.cmap(im.norm(level)) for level in levels][::-1]

    for i, (angles, y_fit, y_errors_fit) in enumerate(
        zip(list_of_angles, list_of_y_fit, list_of_y_errors_fit)
    ):
        axes[1].errorbar(
            angles * units.rad.to(units.deg),
            y_fit,
            yerr=y_errors_fit,
            linestyle="None",
            marker="o",
            markersize=2.5,
            color=colors[i],
        )
    # axes[0].plot(
    #     [247],
    #     [250],
    #     marker="o"
    # )
    xticks = np.linspace(0, image.shape[1], 11)
    yticks = np.linspace(0, image.shape[0], 11)
    axes[0].set_xticks(xticks)
    axes[0].set_yticks(xticks)
    axes[1].set_xticks([0, 90, 180, 270, 360])
    axes[1].set_xlabel(r"$\phi$ (deg)", fontsize=15)
    axes[1].set_ylabel(r"$\rm I(\phi)$ [E/s]", fontsize=15)
    axes[1].tick_params(axis="y", labelsize=12.5)
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")
    for i, ax in enumerate(axes):
        ax.minorticks_on()
        ax.tick_params(
            axis="both",
            which="major",
            length=6,
            right=True,
            top=True,
            direction="in",
            colors="w" if i == 0 else "black",
        )
        ax.tick_params(
            axis="both",
            which="minor",
            length=3,
            right=True,
            top=True,
            direction="in",
            colors="w" if i == 0 else "black",
        )

    axes[1].set_yscale("log")

    # text = axes[0].text(
    #     0.05,
    #     0.95,
    #     "model 1",
    #     horizontalalignment='left',
    #     verticalalignment='center',
    #     transform=axes[0].transAxes,
    #     fontsize=25,
    #     weight="bold",
    #     color="w"
    # )
    # text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='black')])

    # text = axes[0].text(
    #     0.7,
    #     0.95,
    #     "NGC 2274",
    #     horizontalalignment='left',
    #     verticalalignment='center',
    #     transform=axes[0].transAxes,
    #     fontsize=25,
    #     weight="bold",
    #     color="w"
    # )
    # text.set_path_effects([patheffects.withStroke(linewidth=3, foreground='black')])

    if a_max is None:
        axes[0].set_xlim(0, image.shape[1])
        axes[0].set_ylim(0, image.shape[0])
    else:
        x0 = list_of_parameters[0]["x0"]
        y0 = list_of_parameters[0]["y0"]
        axes[0].set_xlim(x0 - a_max, x0 + a_max)
        axes[0].set_ylim(y0 - a_max, y0 + a_max)

    # NOTE:
    figure.subplots_adjust(left=0.05, right=0.95, bottom=0.075, top=0.95, wspace=0.0)

    y = []
    for i, (a, chi_squares_i) in enumerate(zip(array, chi_squares)):
        n = len(chi_squares_i) - 1

        # NOTE:
        y_i = np.nansum(chi_squares_i[:-1]) / (n - m)
        y.append(y_i)

    figure_stats, axes_stats = plt.subplots()
    axes_stats.plot(array, y, linestyle="None", marker="o", color="black")
    axes_stats.set_xscale("log")
    axes_stats.set_yscale("log")
    # directory = "./MASSIVE/metadata"
    # filename = "{}/xy_model_default.numpy".format(directory)
    # with open(filename, 'wb') as f:
    #     np.save(f, [x, y])
    # plt.show()
    # exit()

    # NOTE:
    # chi_squares_flattened = list(itertools.chain(*chi_squares))
    # plt.hist(chi_squares_flattened, bins=100, alpha=0.75)

    # plt.show();exit()
