import autoarray as aa
import autoarray.plot as aplt

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autogalaxy.quantity.fit_quantity import FitQuantity
from autogalaxy.plot.plot_utils import _to_positions


def _make_meta(fit, output_path, output_format, colormap, use_log10, positions, suffix=""):
    from autogalaxy.plot.plot_utils import _resolve_format
    output_format = _resolve_format(output_format)
    output = aplt.Output(path=output_path, format=output_format) if output_path else aplt.Output()
    cmap = aplt.Cmap(cmap=colormap) if colormap != "default" else aplt.Cmap()
    return FitImagingPlotterMeta(
        fit=fit,
        output=output,
        cmap=cmap,
        use_log10=use_log10,
        positions=_to_positions(positions),
    )


def plot_data(
    fit: FitQuantity,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
):
    if isinstance(fit.dataset.data, aa.Array2D):
        _make_meta(fit, output_path, output_format, colormap, use_log10, positions).figures_2d(data=True)
    else:
        _make_meta(fit.y, output_path, output_format, colormap, use_log10, positions).figures_2d(data=True, suffix="_y")
        _make_meta(fit.x, output_path, output_format, colormap, use_log10, positions).figures_2d(data=True, suffix="_x")


def plot_noise_map(
    fit: FitQuantity,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
):
    if isinstance(fit.dataset.data, aa.Array2D):
        _make_meta(fit, output_path, output_format, colormap, use_log10, positions).figures_2d(noise_map=True)
    else:
        _make_meta(fit.y, output_path, output_format, colormap, use_log10, positions).figures_2d(noise_map=True, suffix="_y")
        _make_meta(fit.x, output_path, output_format, colormap, use_log10, positions).figures_2d(noise_map=True, suffix="_x")


def plot_model_image(
    fit: FitQuantity,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
):
    if isinstance(fit.dataset.data, aa.Array2D):
        _make_meta(fit, output_path, output_format, colormap, use_log10, positions).figures_2d(model_image=True)
    else:
        _make_meta(fit.y, output_path, output_format, colormap, use_log10, positions).figures_2d(model_image=True, suffix="_y")
        _make_meta(fit.x, output_path, output_format, colormap, use_log10, positions).figures_2d(model_image=True, suffix="_x")


def plot_residual_map(
    fit: FitQuantity,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
):
    if isinstance(fit.dataset.data, aa.Array2D):
        _make_meta(fit, output_path, output_format, colormap, use_log10, positions).figures_2d(residual_map=True)
    else:
        _make_meta(fit.y, output_path, output_format, colormap, use_log10, positions).figures_2d(residual_map=True, suffix="_y")
        _make_meta(fit.x, output_path, output_format, colormap, use_log10, positions).figures_2d(residual_map=True, suffix="_x")


def plot_normalized_residual_map(
    fit: FitQuantity,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
):
    if isinstance(fit.dataset.data, aa.Array2D):
        _make_meta(fit, output_path, output_format, colormap, use_log10, positions).figures_2d(normalized_residual_map=True)
    else:
        _make_meta(fit.y, output_path, output_format, colormap, use_log10, positions).figures_2d(normalized_residual_map=True, suffix="_y")
        _make_meta(fit.x, output_path, output_format, colormap, use_log10, positions).figures_2d(normalized_residual_map=True, suffix="_x")


def plot_chi_squared_map(
    fit: FitQuantity,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
):
    if isinstance(fit.dataset.data, aa.Array2D):
        _make_meta(fit, output_path, output_format, colormap, use_log10, positions).figures_2d(chi_squared_map=True)
    else:
        _make_meta(fit.y, output_path, output_format, colormap, use_log10, positions).figures_2d(chi_squared_map=True, suffix="_y")
        _make_meta(fit.x, output_path, output_format, colormap, use_log10, positions).figures_2d(chi_squared_map=True, suffix="_x")


def subplot_fit(
    fit: FitQuantity,
    output_path=None,
    output_format="png",
    colormap="default",
    use_log10=False,
    positions=None,
):
    if isinstance(fit.dataset.data, aa.Array2D):
        _make_meta(fit, output_path, output_format, colormap, use_log10, positions).subplot_fit()
    else:
        _make_meta(fit.y, output_path, output_format, colormap, use_log10, positions).subplot_fit()
        _make_meta(fit.x, output_path, output_format, colormap, use_log10, positions).subplot_fit()
