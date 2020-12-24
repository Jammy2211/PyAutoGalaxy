from os import path
import autogalaxy.plot as aplt


class TestLensingMatObj:
    def test__all_load_from_config_correctly(self):

        light_profile_centres_scatter = aplt.LightProfileCentreScatter()

        assert light_profile_centres_scatter.kwargs["size"] == 1

        mass_profile_centres_scatter = aplt.MassProfileCentreScatter()

        assert mass_profile_centres_scatter.kwargs["size"] == 2

        multiple_images_scatter = aplt.MultipleImagesScatter()

        assert multiple_images_scatter.kwargs["size"] == 3

        critical_curves_line = aplt.CriticalCurvesLine()

        assert critical_curves_line.kwargs["width"] == 4

        caustics_line = aplt.CausticsLine()

        assert caustics_line.kwargs["width"] == 5
