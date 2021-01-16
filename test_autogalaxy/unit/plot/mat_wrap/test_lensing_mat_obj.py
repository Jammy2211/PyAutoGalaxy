from os import path
import autogalaxy.plot as aplt


class TestLensingMatObj:
    def test__all_load_from_config_correctly(self):

        light_profile_centres_scatter = aplt.LightProfileCentresScatter()

        assert light_profile_centres_scatter.config_dict["s"] == 1

        mass_profile_centres_scatter = aplt.MassProfileCentresScatter()

        assert mass_profile_centres_scatter.config_dict["s"] == 2

        multiple_images_scatter = aplt.MultipleImagesScatter()

        assert multiple_images_scatter.config_dict["s"] == 3

        critical_curves_plot = aplt.CriticalCurvesPlot()

        assert critical_curves_plot.config_dict["width"] == 4

        caustics_plot = aplt.CausticsPlot()

        assert caustics_plot.config_dict["width"] == 5
