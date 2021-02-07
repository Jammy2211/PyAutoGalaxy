import numpy as np
from autoarray.structures import grids
import typing


class PointSource:
    def __init__(self, centre: typing.Tuple[float, float] = (0.0, 0.0)):

        self.centre = centre


class PointSourceFlux(PointSource):
    def __init__(
        self, centre: typing.Tuple[float, float] = (0.0, 0.0), flux: float = 0.1
    ):

        super().__init__(centre=centre)

        self.flux = flux
