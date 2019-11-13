import autofit as af
import autoastro as aast

import pytest


@pytest.fixture(autouse=True)
def reset_config():
    """
    Use configuration from the default path. You may want to change this to set a specific path.
    """
    af.conf.instance = af.conf.default


class TestCase:
    def test_integration(self):
        # Create a mapper. This can be used to convert values output by a non linear optimiser into class instances.
        mapper = af.ModelMapper()

        # Create a model_galaxy prior for the source model_galaxy. Here we are describing only the light profile of
        # the source model_galaxy which comprises an elliptical exponential and elliptical sersic light profile.
        source_galaxy_prior = aast.GalaxyModel(
            redshift=aast.Redshift,
            light_profile_one=aast.lp.EllipticalExponential,
            light_profile_2=aast.lp.EllipticalSersic,
        )

        # Create a model_galaxy prior for the source model_galaxy. Here we are describing both the light and mass
        # profiles. We've also stipulated that the centres of any galaxies generated using the model_galaxy prior
        # should match.
        lens_galaxy_prior = aast.GalaxyModel(
            redshift=aast.Redshift,
            light_profile=aast.lp.EllipticalExponential,
            mass_profile=aast.mp.EllipticalExponential,
            align_centres=True,
        )

        mapper.source_galaxy = source_galaxy_prior
        mapper.lens_galaxy = lens_galaxy_prior

        # Create a model instance. All the instances of the profile classes are created here. Normally we would do this
        # using the output of a non linear search but in this case we are using the median values from the priors.
        instance = mapper.instance_from_prior_medians()

        # Recover model_galaxy instances. We can pass the model instance to model_galaxy priors to recover a fully
        # constructed model_galaxy
        source_galaxy = instance.source_galaxy
        lens_galaxy = instance.lens_galaxy

        # Let's just check that worked
        assert len(source_galaxy.light_profiles) == 2
        assert len(source_galaxy.mass_profiles) == 0

        assert len(lens_galaxy.light_profiles) == 1
        assert len(lens_galaxy.mass_profiles) == 1

        assert source_galaxy.redshift == 1.5
        assert lens_galaxy.redshift == 1.5
