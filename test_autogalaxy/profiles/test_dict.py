import autogalaxy as ag


def test_to_dict():
    profile = ag.mp.EllSersic()

    assert profile.dict() == {
        "type": "autogalaxy.profiles.mass_profiles.stellar_mass_profiles.EllSersic",
        "centre": (0.0, 0.0),
        "elliptical_comps": (0.0, 0.0),
        "intensity": 0.1,
        "effective_radius": 0.6,
        "sersic_index": 0.6,
        "mass_to_light_ratio": 1.0,
    }
