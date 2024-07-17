import numpy as np
import pytest

import autogalaxy as ag


def test__data(imaging_7x7):

    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.data[0] == pytest.approx(1.0, 1.0e-4)
    assert fit.data[1] == pytest.approx(1.0, 1.0e-4)

def test__noise_map(imaging_7x7):

    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.noise_map[0] == pytest.approx(2.0, 1.0e-4)
    assert fit.noise_map[1] == pytest.approx(2.0, 1.0e-4)


def test__log_likelihood(imaging_7x7):

    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.log_likelihood == pytest.approx(-0.111111111111111, 1.0e-4)