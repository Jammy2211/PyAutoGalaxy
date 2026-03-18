import numpy as np

from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

from autoarray.plot.abstract_plotters import AbstractPlotter

from autogalaxy.plot.mat_plot.one_d import MatPlot1D
from autogalaxy.plot.mat_plot.two_d import MatPlot2D


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


class Plotter(AbstractPlotter):

    def __init__(
        self,
        mat_plot_1d: MatPlot1D = None,
        mat_plot_2d: MatPlot2D = None,
    ):

        super().__init__(
            mat_plot_1d=mat_plot_1d,
            mat_plot_2d=mat_plot_2d,
        )

        self.mat_plot_1d = mat_plot_1d or MatPlot1D()
        self.mat_plot_2d = mat_plot_2d or MatPlot2D()

    def _plot_array(self, array, auto_labels, lines=None, positions=None, grid=None):
        from autoarray.plot.plots.array import plot_array
        from autoarray.structures.plot.structure_plotters import (
            _auto_mask_edge,
            _numpy_lines,
            _numpy_grid,
            _numpy_positions,
            _output_for_mat_plot,
            _zoom_array,
        )

        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d,
            is_sub,
            auto_labels.filename if auto_labels else "array",
        )

        array = _zoom_array(array)

        try:
            arr = array.native.array
            extent = array.geometry.extent
        except AttributeError:
            arr = np.asarray(array)
            extent = None

        mask = _auto_mask_edge(array) if hasattr(array, "mask") else None

        plot_array(
            array=arr,
            ax=ax,
            extent=extent,
            mask=mask,
            grid=_numpy_grid(grid),
            positions=_numpy_positions(positions) if not isinstance(positions, list) else positions,
            lines=_numpy_lines(lines) if not isinstance(lines, list) else lines,
            title=auto_labels.title if auto_labels else "",
            colormap=self.mat_plot_2d.cmap.cmap,
            use_log10=self.mat_plot_2d.use_log10,
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
            structure=array,
        )

    def _plot_grid(self, grid, auto_labels, lines=None):
        from autoarray.plot.plots.grid import plot_grid
        from autoarray.structures.plot.structure_plotters import (
            _output_for_mat_plot,
        )

        is_sub = self.mat_plot_2d.is_for_subplot
        ax = self.mat_plot_2d.setup_subplot() if is_sub else None
        output_path, filename, fmt = _output_for_mat_plot(
            self.mat_plot_2d,
            is_sub,
            auto_labels.filename if auto_labels else "grid",
        )

        plot_grid(
            grid=np.array(grid.array),
            ax=ax,
            title=auto_labels.title if auto_labels else "",
            output_path=output_path,
            output_filename=filename,
            output_format=fmt,
        )
