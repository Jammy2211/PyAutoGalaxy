import os
import numpy as np
import matplotlib.pyplot as plt

from autoarray.plot.abstract_plotters import AbstractPlotter
from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap


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


def _save_subplot(fig, output, auto_filename):
    """Save or show a subplot figure using an Output object."""
    from autoarray.structures.plot.structure_plotters import _output_for_plotter

    output_path, filename, fmt = _output_for_plotter(output, auto_filename)
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        fig.savefig(
            os.path.join(output_path, f"{filename}.{fmt}"),
            bbox_inches="tight",
            pad_inches=0.1,
        )
    else:
        plt.show()
    plt.close(fig)


class Plotter(AbstractPlotter):

    def __init__(
        self,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        title: str = None,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10, title=title)

    def _plot_array(self, array, auto_filename, title, lines=None, positions=None, grid=None, ax=None):
        from autoarray.plot.plots.array import plot_array
        from autoarray.structures.plot.structure_plotters import (
            _auto_mask_edge,
            _numpy_lines,
            _numpy_grid,
            _numpy_positions,
            _output_for_plotter,
            _zoom_array,
        )

        if ax is None:
            output_path, filename, fmt = _output_for_plotter(self.output, auto_filename)
        else:
            output_path, filename, fmt = None, auto_filename, "png"

        array = _zoom_array(array)

        try:
            arr = array.native.array
            extent = array.geometry.extent
        except AttributeError:
            arr = np.asarray(array)
            extent = None

        mask = _auto_mask_edge(array) if hasattr(array, "mask") else None

        _positions = positions if isinstance(positions, list) else _numpy_positions(positions)
        _lines = lines if isinstance(lines, list) else _numpy_lines(lines)

        plot_array(
            array=arr,
            ax=ax,
            extent=extent,
            mask=mask,
            grid=_numpy_grid(grid),
            positions=_positions,
            lines=_lines,
            title=title or "",
            colormap=self.cmap.cmap,
            use_log10=self.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
            structure=array,
        )

    def _plot_grid(self, grid, auto_filename, title, lines=None, ax=None):
        from autoarray.plot.plots.grid import plot_grid
        from autoarray.structures.plot.structure_plotters import _output_for_plotter

        if ax is None:
            output_path, filename, fmt = _output_for_plotter(self.output, auto_filename)
        else:
            output_path, filename, fmt = None, auto_filename, "png"

        plot_grid(
            grid=np.array(grid.array),
            ax=ax,
            title=title or "",
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )
