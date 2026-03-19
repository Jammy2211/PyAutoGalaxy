import matplotlib.pyplot as plt

from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap

import autoarray as aa

from autogalaxy.profiles.light.abstract import LightProfile
from autogalaxy.profiles.basis import Basis
from autogalaxy.plot.abstract_plotters import Plotter, _to_positions, _save_subplot
from autogalaxy.profiles.plot.light_profile_plotters import LightProfilePlotter
from autogalaxy import exc


class BasisPlotter(Plotter):
    def __init__(
        self,
        basis: Basis,
        grid: aa.type.Grid1D2DLike,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        positions=None,
        lines=None,
    ):
        from autogalaxy.profiles.light.linear import LightProfileLinear

        for light_profile in basis.light_profile_list:
            if isinstance(light_profile, LightProfileLinear):
                raise exc.raise_linear_light_profile_in_plot(
                    plotter_type=self.__class__.__name__,
                )

        self.basis = basis
        self.grid = grid
        self.positions = positions
        self.lines = lines

        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

    def light_profile_plotter_from(self, light_profile: LightProfile) -> LightProfilePlotter:
        return LightProfilePlotter(
            light_profile=light_profile,
            grid=self.grid,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            half_light_radius=light_profile.half_light_radius,
        )

    def subplot_image(self):
        n = len(self.basis.light_profile_list)
        cols = min(n, 4)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 7 * rows))
        import numpy as np
        axes_flat = [axes] if n == 1 else list(np.array(axes).flatten())

        positions = _to_positions(self.positions)

        for i, light_profile in enumerate(self.basis.light_profile_list):
            self._plot_array(
                array=light_profile.image_2d_from(grid=self.grid),
                auto_filename="subplot_basis_image",
                title=light_profile.coefficient_tag,
                positions=positions,
                lines=self.lines,
                ax=axes_flat[i],
            )

        plt.tight_layout()
        _save_subplot(fig, self.output, "subplot_basis_image")
