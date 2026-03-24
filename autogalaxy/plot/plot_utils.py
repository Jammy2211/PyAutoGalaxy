import logging
import os
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _to_lines(*items):
    """Convert multiple line sources into a flat list of (N,2) numpy arrays."""
    result = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, list):
            for sub in item:
                try:
                    arr = np.array(sub.array if hasattr(sub, "array") else sub)
                    if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                        result.append(arr)
                except Exception:
                    pass
        else:
            try:
                arr = np.array(item.array if hasattr(item, "array") else item)
                if arr.ndim == 2 and arr.shape[1] == 2 and len(arr) > 0:
                    result.append(arr)
            except Exception:
                pass
    return result or None


def _to_positions(*items):
    """Convert multiple position sources into a flat list of (N,2) numpy arrays."""
    return _to_lines(*items)


def _save_subplot(fig, output_path, output_filename, output_format="png",
                  dpi=300, structure=None):
    """Save a subplot figure to disk (or show it if output_path is falsy).

    Mirrors the interface of ``autoarray.plot.plots.utils.save_figure``.
    When ``output_format`` is ``"fits"`` the *structure* argument is used to
    write a FITS file via its ``output_to_fits`` method.
    """
    fmt = output_format[0] if isinstance(output_format, (list, tuple)) else (output_format or "png")
    if output_path:
        os.makedirs(str(output_path), exist_ok=True)
        fpath = os.path.join(str(output_path), f"{output_filename}.{fmt}")
        if fmt == "fits":
            if structure is not None and hasattr(structure, "output_to_fits"):
                structure.output_to_fits(file_path=fpath, overwrite=True)
            else:
                logger.warning(
                    f"_save_subplot: fits format requested for {output_filename} "
                    "but no compatible structure was provided; skipping."
                )
        else:
            fig.savefig(fpath, dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()
    plt.close(fig)


def _resolve_colormap(colormap):
    """Resolve 'default' to the autoarray default colormap."""
    if colormap == "default":
        return "jet"
    return colormap


def _resolve_format(output_format):
    """Normalise output_format: accept a list/tuple or a plain string."""
    if isinstance(output_format, (list, tuple)):
        return output_format[0]
    return output_format or "png"


def _numpy_grid(grid):
    """Convert a grid-like object to a numpy array, or return None."""
    if grid is None:
        return None
    try:
        return np.array(grid.array if hasattr(grid, "array") else grid)
    except Exception:
        return None


def plot_array(
    array,
    title,
    output_path=None,
    output_filename="array",
    output_format="png",
    colormap="default",
    use_log10=False,
    vmin=None,
    vmax=None,
    symmetric=False,
    positions=None,
    lines=None,
    grid=None,
    ax=None,
):
    """Plot an autoarray Array2D to file or onto an existing Axes.

    All array preprocessing (zoom, mask-edge extraction, native/extent
    unpacking) is handled internally so callers never need to duplicate it.
    """
    from autoarray.plot import plot_array as _aa_plot_array
    from autoarray.plot import zoom_array, auto_mask_edge

    colormap = _resolve_colormap(colormap)
    output_format = _resolve_format(output_format)
    array = zoom_array(array)

    try:
        arr = array.native.array
        extent = array.geometry.extent
    except AttributeError:
        arr = np.asarray(array)
        extent = None

    mask = auto_mask_edge(array) if hasattr(array, "mask") else None

    if symmetric:
        finite = arr[np.isfinite(arr)]
        abs_max = float(np.max(np.abs(finite))) if len(finite) > 0 else 1.0
        vmin, vmax = -abs_max, abs_max

    _positions_list = positions if isinstance(positions, list) else _to_positions(positions)
    _lines_list = lines if isinstance(lines, list) else _to_lines(lines)

    _output_path = None if ax is not None else output_path

    _aa_plot_array(
        array=arr,
        ax=ax,
        extent=extent,
        mask=mask,
        grid=_numpy_grid(grid),
        positions=_positions_list,
        lines=_lines_list,
        title=title or "",
        colormap=colormap,
        use_log10=use_log10,
        vmin=vmin,
        vmax=vmax,
        output_path=_output_path,
        output_filename=output_filename,
        output_format=output_format,
        structure=array,
    )


def plot_grid(
    grid,
    title,
    output_path=None,
    output_filename="grid",
    output_format="png",
    lines=None,
    ax=None,
):
    """Plot an autoarray Grid2D to file or onto an existing Axes."""
    from autoarray.plot import plot_grid as _aa_plot_grid

    output_format = _resolve_format(output_format)
    _output_path = None if ax is not None else output_path

    _aa_plot_grid(
        grid=np.array(grid.array),
        ax=ax,
        title=title or "",
        output_path=_output_path,
        output_filename=output_filename,
        output_format=output_format,
    )


def _critical_curves_from(mass_obj, grid, tc=None, rc=None):
    """Compute tangential and radial critical curves for a mass object."""
    from autogalaxy.operate.lens_calc import LensCalc

    if tc is None:
        od = LensCalc.from_mass_obj(mass_obj)
        tc = od.tangential_critical_curve_list_from(grid=grid)
        rc_area = od.radial_critical_curve_area_list_from(grid=grid)
        if any(area > grid.pixel_scale for area in rc_area):
            rc = od.radial_critical_curve_list_from(grid=grid)

    return tc, rc
