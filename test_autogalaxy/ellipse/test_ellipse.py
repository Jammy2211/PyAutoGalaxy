import numpy as np
import pytest

import autogalaxy as ag

def test__ellipticity():

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0))

    assert ellipse.ellipticity == 0.0

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.5))

    assert ellipse.ellipticity == pytest.approx(0.5, 1.0e-4)

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.0))

    assert ellipse.ellipticity == pytest.approx(0.5, 1.0e-4)

    ellipse = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5))

    assert ellipse.ellipticity == pytest.approx(np.sqrt(2)/2.0, 1.0e-4)