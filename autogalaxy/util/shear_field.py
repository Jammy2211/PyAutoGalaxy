import logging
from matplotlib.patches import Ellipse
import numpy as np
import typing

import autoarray as aa

logging.basicConfig()
logger = logging.getLogger(__name__)


class ShearField2DIrregular(aa.VectorField2DIrregular):
    def __new__(
        cls, vectors: np.ndarray or [(float, float)], grid: aa.Grid2DIrregular or list
    ):
        """
        An irregular shear field, which is collection of (y,x) vectors which are located on an irregular grid
        of (y,x) coordinates.

        The structure of this data structure is described in
        `autoarray.structures.vector_fields.vector_field_irregular.VectorField2DIrregular`

        This class extends `VectorField2DIrregular` to include methods that are specific to a shear field, typically
        used for weak lensing calculations.

        Parameters
        ----------
        vectors
            The 2D (y,x) vectors on an irregular grid that represent the vector-field.
        grid
            The irregular grid of (y,x) coordinates where each vector is located.
        """

        if len(vectors) == 0:
            return []

        if type(vectors) is list:
            vectors = np.asarray(vectors)

        obj = vectors.view(cls)
        obj.grid = aa.Grid2DIrregular(grid=grid)

        return obj

    @property
    def ellipticities(self) -> aa.ValuesIrregular:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the galaxy ellipticity each vector
        corresponds too.
        """
        return aa.ValuesIrregular(values=np.sqrt(self[:, 0] ** 2 + self[:, 1] ** 2.0))

    @property
    def semi_major_axes(self) -> aa.ValuesIrregular:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the semi-major axis of each
        galaxy ellipticity that each vector corresponds too.
        """
        return aa.ValuesIrregular(values=3 * (1 + self.ellipticities))

    @property
    def semi_minor_axes(self) -> aa.ValuesIrregular:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the semi-minor axis of each
        galaxy ellipticity that each vector corresponds too.
        """
        return aa.ValuesIrregular(values=3 * (1 - self.ellipticities))

    @property
    def phis(self) -> aa.ValuesIrregular:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the position angle defined
        counter clockwise from the positive x-axis of each galaxy ellipticity that each vector corresponds too.
        """
        return aa.ValuesIrregular(
            values=np.arctan2(self[:, 0], self[:, 1]) * 180.0 / np.pi / 2.0
        )

    @property
    def elliptical_patches(self) -> typing.List[Ellipse]:
        """
        If we treat this vector field as a set of weak lensing shear measurements, the elliptical patch representing
        each galaxy ellipticity. This patch is used for visualizing an ellipse of each galaxy in an image.
        """

        return [
            Ellipse(
                xy=(x, y), width=semi_major_axis, height=semi_minor_axis, angle=angle
            )
            for x, y, semi_major_axis, semi_minor_axis, angle in zip(
                self.grid[:, 1],
                self.grid[:, 0],
                self.semi_major_axes,
                self.semi_minor_axes,
                self.phis,
            )
        ]
