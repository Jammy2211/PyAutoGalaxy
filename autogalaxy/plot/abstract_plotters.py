from autoarray.plot.wrap.base.abstract import set_backend

set_backend()

from autoarray.plot.abstract_plotters import AbstractPlotter


class Plotter(AbstractPlotter):
    @property
    def get_1d(self):
        from autogalaxy.plot.get_visuals.one_d import GetVisuals1D

        return GetVisuals1D(visuals=self.visuals_1d, include=self.include_1d)

    @property
    def get_2d(self):
        from autogalaxy.plot.get_visuals.two_d import GetVisuals2D

        return GetVisuals2D(visuals=self.visuals_2d, include=self.include_2d)
