import numpy as np
import pytest

import autogalaxy as ag

mask = ag.Mask2D(
    mask=[
        [True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True],
        [True, True, False, False, False, True, True],
        [True, True, False, True, False, True, True],
        [True, True, False, False, False, True, True],
        [True, True, True, True, True, True, True],
        [True, True, True, True, True, True, True],
    ],
    pixel_scales=1.0,
)


def test__mask_interp(imaging_7x7):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.mask_interp[0] == pytest.approx(False, 1.0e-4)
    assert fit.mask_interp[1] == pytest.approx(False, 1.0e-4)

    imaging_7x7 = imaging_7x7.apply_mask(mask=mask)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.mask_interp[0] == pytest.approx(False, 1.0e-4)
    assert fit.mask_interp[1] == pytest.approx(True, 1.0e-4)


def test__data_interp(imaging_7x7):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.data_interp[0] == pytest.approx(1.0, 1.0e-4)
    assert fit.data_interp[1] == pytest.approx(1.0, 1.0e-4)

    imaging_7x7 = imaging_7x7.apply_mask(mask=mask)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.data_interp[0] == pytest.approx(1.0, 1.0e-4)
    assert np.isnan(fit.data_interp[1])

def test__noise_map_interp(imaging_7x7):
    ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.noise_map_interp[0] == pytest.approx(2.0, 1.0e-4)
    assert fit.noise_map_interp[1] == pytest.approx(2.0, 1.0e-4)

    imaging_7x7 = imaging_7x7.apply_mask(mask=mask)

    fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)

    assert fit.noise_map_interp[0] == pytest.approx(2.0, 1.0e-4)
    assert np.isnan(fit.noise_map_interp[1])

#
# def test__total_points_interp(imaging_7x7):
#
#     ellipse_0 = ag.Ellipse(centre=(0.0, 0.0), ell_comps=(0.0, 0.0), major_axis=1.0)
#
#     fit = ag.FitEllipse(dataset=imaging_7x7, ellipse=ellipse_0)
#
#     assert fit.total_points_interp == 6
#
#     mask = np.array(
#         [
#             [True, True, True, True, True, True, True],
#             [True, True, True, True, True, True, True],
#             [True, True, True, True, True, True, True],
#             [True, True, False, False, False, True, True],
#             [True, True, False, False, False, True, True],
#             [True, True, True, True, True, True, True],
#             [True, True, True, True, True, True, True],
#         ]
#     )
#
#     mask = ag.Mask2D(mask=mask, pixel_scales=(1.0, 1.0))
#
#     imaging = imaging_7x7.apply_mask(mask=mask)
#
#     fit = ag.FitEllipse(dataset=imaging, ellipse=ellipse_0)
#
#     assert fit.total_points_interp == 4


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
