import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.plot.plot_utils import _to_positions, plot_array
from autogalaxy import exc


def plot_image_2d(
    light_profile: LightProfile,
    grid: aa.type.Grid1D2DLike,
    output_path=None,
    output_filename="image_2d",
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
    lines=None,
    ax=None,
):
    from autogalaxy.profiles.light.linear import LightProfileLinear

    if isinstance(light_profile, LightProfileLinear):
        raise exc.raise_linear_light_profile_in_plot(
            plotter_type="plot_image_2d",
        )

    plot_array(
        array=light_profile.image_2d_from(grid=grid),
        title="Image",
        output_path=output_path,
        output_filename=output_filename,
        output_format=output_format,
        colormap=colormap,
        use_log10=use_log10,
        positions=_to_positions(positions),
        lines=lines,
        ax=ax,
    )
