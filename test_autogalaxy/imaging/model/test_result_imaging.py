import numpy as np
import pytest

import autofit as af
import autogalaxy as ag

from autogalaxy.imaging.model.result import ResultImaging


def test__results_include_masked_dataset_and_mask(
    analysis_imaging_7x7, masked_imaging_7x7, samples_with_result
):

    result = ResultImaging(
        samples=samples_with_result, analysis=analysis_imaging_7x7, model=None
    )

    assert (result.mask == masked_imaging_7x7.mask).all()
    assert (result.dataset.image == masked_imaging_7x7.image).all()


def test___image_dict(analysis_imaging_7x7):

    galaxies = af.ModelInstance()
    galaxies.galaxy = ag.Galaxy(redshift=0.5)
    galaxies.source = ag.Galaxy(redshift=0.5)

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    samples = ag.m.MockSamples(max_log_likelihood_instance=instance)

    result = ResultImaging(samples=samples, analysis=analysis_imaging_7x7, model=None)

    image_dict = result.image_galaxy_dict
    assert isinstance(image_dict[("galaxies", "galaxy")], np.ndarray)
    assert isinstance(image_dict[("galaxies", "source")], np.ndarray)

    result.instance.galaxies.light = ag.Galaxy(redshift=0.5)

    image_dict = result.image_galaxy_dict
    assert (image_dict[("galaxies", "galaxy")].native == np.zeros((7, 7))).all()
    assert isinstance(image_dict[("galaxies", "source")], np.ndarray)


def test___linear_light_profiles_in_result(analysis_imaging_7x7):

    galaxies = af.ModelInstance()
    galaxies.galaxy = ag.Galaxy(redshift=0.5, bulge=ag.lp_linear.EllSersic())

    instance = af.ModelInstance()
    instance.galaxies = galaxies

    samples = ag.m.MockSamples(max_log_likelihood_instance=instance)

    result = ResultImaging(samples=samples, analysis=analysis_imaging_7x7, model=None)

    assert not isinstance(
        result.max_log_likelihood_plane.galaxies[0].bulge,
        ag.lp_linear.LightProfileLinear,
    )
    assert result.max_log_likelihood_plane.galaxies[0].bulge.intensity == pytest.approx(
        0.0054343, 1.0e-4
    )
