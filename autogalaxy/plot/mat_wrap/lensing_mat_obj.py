from autoarray.plot.mat_wrap import mat_obj, mat_structure


class LightProfileCentresScatter(mat_obj.AbstractMatObj, mat_structure.GridScatter):
    pass


class MassProfileCentresScatter(mat_obj.AbstractMatObj, mat_structure.GridScatter):
    pass


class MultipleImagesScatter(mat_obj.AbstractMatObj, mat_structure.GridScatter):
    pass


class CriticalCurvesPlot(mat_obj.AbstractMatObj, mat_structure.LinePlot):
    pass


class CausticsPlot(mat_obj.AbstractMatObj, mat_structure.LinePlot):
    pass
