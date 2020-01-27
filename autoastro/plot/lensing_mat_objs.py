from autoarray.plot import mat_objs


class LightProfileCentreScatterer(mat_objs.Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):

        super(LightProfileCentreScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="light_profile_centres",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return LightProfileCentreScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class MassProfileCentreScatterer(mat_objs.Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):

        super(MassProfileCentreScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="mass_profile_centres",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return MassProfileCentreScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class MultipleImagesScatterer(mat_objs.Scatterer):
    def __init__(self, size=None, marker=None, colors=None, from_subplot_config=False):

        super(MultipleImagesScatterer, self).__init__(
            size=size,
            marker=marker,
            colors=colors,
            section="multiple_images",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, size=None, marker=None, colors=None):
        return MultipleImagesScatterer(
            size=size, marker=marker, colors=colors, from_subplot_config=True
        )


class CriticalCurvesLiner(mat_objs.Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(CriticalCurvesLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="critical_curves",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return CriticalCurvesLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )


class CausticsLiner(mat_objs.Liner):
    def __init__(
        self,
        width=None,
        style=None,
        colors=None,
        pointsize=None,
        from_subplot_config=False,
    ):

        super(CausticsLiner, self).__init__(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            section="caustics",
            from_subplot_config=from_subplot_config,
        )

    @classmethod
    def sub(cls, width=None, style=None, colors=None, pointsize=None):
        return CausticsLiner(
            width=width,
            style=style,
            colors=colors,
            pointsize=pointsize,
            from_subplot_config=True,
        )
