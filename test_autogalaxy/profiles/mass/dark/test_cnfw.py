import numpy as np
import pytest

import autogalaxy as ag

def test__deflections_yx_2d_from():
    cnfw = ag.mp.cNFWsph(centre=(0.0, 0.0), kappa_s=0.01591814312464436, scale_radius=0.36, core_radius=0.036)

    deflection_2d = cnfw.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflection_r = np.sqrt(deflection_2d[0, 0]**2 + deflection_2d[0, 1]**2)

    assert deflection_r == pytest.approx(0.006034319441107217, 1.0e-8)

def test_convergence_2d_from():
    cnfw = ag.mp.cNFWsph(centre=(0.0, 0.0), kappa_s=0.01591814312464436, scale_radius=0.36, core_radius=0.036)

    convergence = cnfw.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert convergence == pytest.approx(0.0, 1.0e-4)

def test_potential_2d_from():
    cnfw = ag.mp.cNFWsph(centre=(0.0, 0.0), kappa_s=0.01591814312464436, scale_radius=0.36, core_radius=0.036)

    potential = cnfw.potential_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))

    assert potential == pytest.approx(0.0, 1.0e-4)