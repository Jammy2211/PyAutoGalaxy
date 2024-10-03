import numpy as np
import pytest

import autogalaxy as ag


def test__data_interp(imaging_7x7):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.data_interp[0] == pytest.approx(1.0, 1.0e-4)
    assert fit.data_interp[1] == pytest.approx(1.0, 1.0e-4)


def test__noise_map_interp(imaging_7x7):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.noise_map_interp[0] == pytest.approx(2.0, 1.0e-4)
    assert fit.noise_map_interp[1] == pytest.approx(2.0, 1.0e-4)


def test__log_likelihood(imaging_7x7):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.log_likelihood == pytest.approx(0.0, 1.0e-4)


def test__points_from_major_axis(imaging_7x7):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit._points_from_major_axis[1, 0] == pytest.approx(-0.21232, 1.0e-4)
    assert fit._points_from_major_axis[1, 1] == pytest.approx(0.068987, 1.0e-4)


def test___points_from_major_axis__multipole(imaging_7x7):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.5, 0.5), major_axis=1.0)

    multipole = ag.EllipseMultipole(m=4, multipole_comps=(0.2, 0.3))

    fit = ag.FitEllipse(
        dataset=imaging_7x7, ellipse=ellipse_0, multipole_list=[multipole]
    )

    assert fit._points_from_major_axis[1, 0] == pytest.approx(-0.542453, 1.0e-4)
    assert fit._points_from_major_axis[1, 1] == pytest.approx(-0.038278334, 1.0e-4)
