import os
import numpy as np
import matplotlib.pyplot as plt


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


def _save_subplot(fig, output_path, output_filename, output_format="png"):
    """Save a subplot figure to disk or display it."""
    # Normalise format: accept a list (e.g. ['png']) or a plain string
    if isinstance(output_format, (list, tuple)):
        output_format = output_format[0]
    if output_path:
        os.makedirs(str(output_path), exist_ok=True)
        fig.savefig(
            os.path.join(str(output_path), f"{output_filename}.{output_format}"),
            bbox_inches="tight",
            pad_inches=0.1,
        )
    else:
        plt.show()
    plt.close(fig)


def _resolve_colormap(colormap):
    """Resolve 'default' to the actual default matplotlib colormap from Cmap."""
    if colormap == "default":
        from autoarray.plot.wrap.base.cmap import Cmap
        return Cmap().cmap
    return colormap


def _resolve_format(output_format):
    """Normalise output_format: accept a list/tuple or a plain string."""
    if isinstance(output_format, (list, tuple)):
        return output_format[0]
    return output_format or "png"


def _zoom_array(array):
    """Apply zoom_around_mask if configured; otherwise return unchanged."""
    try:
        from autoconf import conf
        zoom = conf.instance["visualize"]["general"]["general"]["zoom_around_mask"]
    except Exception:
        zoom = False
    if zoom and hasattr(array, "mask") and not array.mask.is_all_false:
        try:
            from autoarray.mask.derive.zoom_2d import Zoom2D
            return Zoom2D(mask=array.mask).array_2d_from(array=array, buffer=1)
        except Exception:
            pass
    return array


def _auto_mask_edge(array):
    """Return edge-pixel coordinates of the array's mask, or None."""
    try:
        if not array.mask.is_all_false:
            return np.array(array.mask.derive_grid.edge.array)
    except Exception:
        pass
    return None


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
    positions=None,
    lines=None,
    grid=None,
    ax=None,
):
    """Plot an autoarray Array2D to file or onto an existing Axes."""
    from autoarray.plot.plots.array import plot_array as _aa_plot_array

    colormap = _resolve_colormap(colormap)
    output_format = _resolve_format(output_format)
    array = _zoom_array(array)

    try:
        arr = array.native.array
        extent = array.geometry.extent
    except AttributeError:
        arr = np.asarray(array)
        extent = None

    mask = _auto_mask_edge(array) if hasattr(array, "mask") else None

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
    from autoarray.plot.plots.grid import plot_grid as _aa_plot_grid

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
