import pytest
from os import path
import os
import shutil

import autofit as af
import autogalaxy as ag
from autofit.non_linear.samples import Sample


def clean(database_file, result_path):
    if path.exists(database_file):
        os.remove(database_file)

    if path.exists(result_path):
        shutil.rmtree(result_path)


@pytest.fixture(name="model")
def make_model():
    return af.Collection(
        galaxies=af.Collection(
            lens=af.Model(ag.Galaxy, redshift=0.5, light=ag.lp.Sersic),
            source=af.Model(ag.Galaxy, redshift=1.0, light=ag.lp.Sersic),
        )
    )


@pytest.fixture(name="samples")
def make_samples(model):
    galaxy_0 = ag.Galaxy(redshift=0.5, light=ag.lp.Sersic(centre=(0.0, 1.0)))
    galaxy_1 = ag.Galaxy(redshift=1.0, light=ag.lp.Sersic())

    plane = ag.Plane(galaxies=[galaxy_0, galaxy_1])

    parameters = [model.prior_count * [1.0], model.prior_count * [10.0]]

    sample_list = Sample.from_lists(
        model=model,
        parameter_lists=parameters,
        log_likelihood_list=[1.0, 2.0],
        log_prior_list=[0.0, 0.0],
        weight_list=[0.0, 1.0],
    )

    return ag.m.MockSamples(
        model=model,
        sample_list=sample_list,
        max_log_likelihood_instance=plane,
        gaussian_tuples=[(1.0, 2.0)] * model.prior_count,
    )
