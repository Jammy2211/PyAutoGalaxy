import numpy as np
import pytest

import autogalaxy as ag

def test__deflections_yx_2d_from():
    cnfw = ag.mp.cNFW(centre=(0.0, 0.0), kappa_s=0.01591814312464436, scale_radius=0.36, core_radius=0.036)

    deflection_2d = cnfw.deflections_yx_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    deflection_r = np.sqrt(deflection_2d[0]**2 + deflection_2d[1]**2)

    assert deflection_r == pytest.approx(0.006034319441107217, 1.0e-8)

def test_convergence_2d_from():
    cnfw = ag.mp.cNFW(centre=(0.0, 0.0), kappa_s=0.01591814312464436, scale_radius=0.36, core_radius=0.036)

    convergence_2d = cnfw.convergence_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    convergence_r = np.sqrt(convergence_2d[0]**2 + convergence_2d[1]**2)

    assert convergence_r == pytest.approx(0.0, 1.0e-4)

def potential_2d_from():
    cnfw = ag.mp.cNFW(centre=(0.0, 0.0), kappa_s=0.01591814312464436, scale_radius=0.36, core_radius=0.036)

    potential_2d = cnfw.potential_2d_from(grid=ag.Grid2DIrregular([[1.0, 0.0]]))
    potential_r = np.sqrt(potential_2d[0] ** 2 + potential_2d[1] ** 2)

    assert potential_r == pytest.approx(0.0, 1.0e-4)