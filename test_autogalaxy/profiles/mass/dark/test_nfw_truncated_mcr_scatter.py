import numpy as np
import pytest

import autogalaxy as ag

grid = ag.Grid2DIrregular([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [2.0, 4.0]])


def test__scatter_is_nonzero():
    mp = ag.mp.NFWTruncatedMCRScatterLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        scatter_sigma=1.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mp.scale_radius == pytest.approx(0.14978, 1.0e-4)
    assert mp.truncation_radius == pytest.approx(33.7134116, 1.0e-4)

    mp = ag.mp.NFWTruncatedMCRScatterLudlowSph(
        centre=(1.0, 2.0),
        mass_at_200=1.0e9,
        scatter_sigma=-1.0,
        redshift_object=0.6,
        redshift_source=2.5,
    )

    # We uare using the NFWTruncatedSph to check the mass gives a conosistnt kappa_s, given certain radii.

    assert mp.scale_radius == pytest.approx(0.29886, 1.0e-4)
    assert mp.truncation_radius == pytest.approx(33.7134116, 1.0e-4)
