import autogalaxy.plot as aplt


def test__mat_obj__all_load_from_config_correctly():
    aplt.LightProfileCentresScatter()
    aplt.MassProfileCentresScatter()
    aplt.MultipleImagesScatter()
    aplt.TangentialCriticalCurvesPlot()
    aplt.TangentialCausticsPlot()
    aplt.RadialCriticalCurvesPlot()
    aplt.RadialCausticsPlot()
