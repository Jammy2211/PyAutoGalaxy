from autoarray.plot.mat_wrap import mat_obj, mat_structure


class LightProfileCentreScatter(mat_obj.AbstractMatObj, mat_structure.GridScatter):
    pass


class MassProfileCentreScatter(mat_obj.AbstractMatObj, mat_structure.GridScatter):
    pass


class MultipleImagesScatter(mat_obj.AbstractMatObj, mat_structure.GridScatter):
    pass


class CriticalCurvesLine(mat_obj.AbstractMatObj, mat_structure.LinePlot):
    pass


class CausticsLine(mat_obj.AbstractMatObj, mat_structure.LinePlot):
    pass
