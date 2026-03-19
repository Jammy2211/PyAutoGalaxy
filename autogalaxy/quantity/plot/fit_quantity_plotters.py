from autoarray.plot.wrap.base.output import Output
from autoarray.plot.wrap.base.cmap import Cmap

import autoarray as aa

from autoarray.fit.plot.fit_imaging_plotters import FitImagingPlotterMeta

from autogalaxy.quantity.fit_quantity import FitQuantity
from autogalaxy.plot.abstract_plotters import Plotter, _to_positions


class FitQuantityPlotter(Plotter):
    def __init__(
        self,
        fit: FitQuantity,
        output: Output = None,
        cmap: Cmap = None,
        use_log10: bool = False,
        positions=None,
    ):
        super().__init__(output=output, cmap=cmap, use_log10=use_log10)

        self.fit = fit
        self.positions = positions

    def _make_positions(self):
        return _to_positions(self.positions)

    def _meta_plotter(self, fit):
        return FitImagingPlotterMeta(
            fit=fit,
            output=self.output,
            cmap=self.cmap,
            use_log10=self.use_log10,
            positions=self._make_positions(),
        )

    def figures_2d(
        self,
        image: bool = False,
        noise_map: bool = False,
        signal_to_noise_map: bool = False,
        model_image: bool = False,
        residual_map: bool = False,
        normalized_residual_map: bool = False,
        chi_squared_map: bool = False,
    ):
        kwargs = dict(
            data=image,
            noise_map=noise_map,
            signal_to_noise_map=signal_to_noise_map,
            model_image=model_image,
            residual_map=residual_map,
            normalized_residual_map=normalized_residual_map,
            chi_squared_map=chi_squared_map,
        )

        if isinstance(self.fit.dataset.data, aa.Array2D):
            self._meta_plotter(self.fit).figures_2d(**kwargs)
        else:
            self._meta_plotter(self.fit.y).figures_2d(**{k: v for k, v in kwargs.items()}, suffix="_y")
            self._meta_plotter(self.fit.x).figures_2d(**{k: v for k, v in kwargs.items()}, suffix="_x")

    def subplot_fit(self):
        if isinstance(self.fit.dataset.data, aa.Array2D):
            self._meta_plotter(self.fit).subplot_fit()
        else:
            self._meta_plotter(self.fit.y).subplot_fit()
            self._meta_plotter(self.fit.x).subplot_fit()
