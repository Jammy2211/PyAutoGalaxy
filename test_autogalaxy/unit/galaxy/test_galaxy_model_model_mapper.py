from autoconf import conf
import autofit as af
import autogalaxy as ag

import pytest


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    conf.instance = conf.default


class TestCase:
    def test_integration(self):
        # Create a mapper. This can be used to convert values output by a non linear optimiser into class instances.
        mapper = af.ModelMapper()

        # Create a model_galaxy prior for the source model_galaxy. Here we are describing only the light profile of
        # the source model_galaxy which comprises an elliptical exponential and elliptical sersic light profile.
        source_galaxy_prior = ag.GalaxyModel(
            redshift=ag.Redshift,
            light_profile_one=ag.lp.EllipticalExponential,
            light_profile_2=ag.lp.EllipticalSersic,
        )

        # Create a model_galaxy prior for the source model_galaxy. Here we are describing both the light and mass
        # profiles. We've also stipulated that the centres of any galaxies generated using the model_galaxy prior
        # should match.
        galaxy_galaxy_prior = ag.GalaxyModel(
            redshift=ag.Redshift,
            light_profile=ag.lp.EllipticalExponential,
            mass_profile=ag.mp.EllipticalExponential,
            align_centres=True,
        )

        mapper.galaxy_1 = source_galaxy_prior
        mapper.galaxy_0 = galaxy_galaxy_prior

        # Create a model instance. All the instances of the profile classes are created here. Normally we would do this
        # using the output of a non linear search but in this case we are using the median values from the priors.
        instance = mapper.instance_from_prior_medians()

        # Recover model_galaxy instances. We can pass the model instance to model_galaxy priors to recover a fully
        # constructed model_galaxy
        galaxy_1 = instance.galaxy_1
        galaxy_0 = instance.galaxy_0

        # Let's just check that worked
        assert len(galaxy_1.light_profiles) == 2
        assert len(galaxy_1.mass_profiles) == 0

        assert len(galaxy_0.light_profiles) == 1
        assert len(galaxy_0.mass_profiles) == 1

        assert galaxy_1.redshift == 1.5
        assert galaxy_0.redshift == 1.5
