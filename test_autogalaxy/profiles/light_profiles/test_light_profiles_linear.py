import autogalaxy as ag

from autogalaxy.profiles.light_profiles.light_profiles_linear import LightProfileLinear


def test__lp_from():

    lp_linear = ag.lp_linear.EllSersic(centre=(1.0, 2.0))

    lp_non_linear = lp_linear.lp_instance_from(intensity=3.0)

    assert not isinstance(lp_non_linear, LightProfileLinear)
    assert type(lp_non_linear) is ag.lp.EllSersic
    assert lp_non_linear.centre == (1.0, 2.0)
    assert lp_non_linear.intensity == 3.0
