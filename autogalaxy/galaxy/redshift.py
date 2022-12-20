class Redshift(float):
    """
    Class used when assigning a redshift to a `Galaxy` object.

    This object is only required when making the `Redshift` of the `Galaxy` a free parameter in a model that
    is fitted.

    This is because **PyAutoFit** (which handles model-fitting), requires all parameters to be a Python class.

    The `Redshift` object does not need to be used for general **PyAutoGalaxy** use.

    Examples
    --------

    import autofit as af
    import autogalaxy as ag

    bulge = af.Model(ag.lp.Sersic)

    redshift = af.Model(ag.Redshift)
    redshift.redshift = af.UniformPrior(lower_limit=0.0, upper_limit=2.0)

    galaxy = af.Model(ag.Galaxy, redshift=redshift, bulge=bulge)

    model = af.Collection(galaxies=af.Collection(galaxy=galaxy))
    """

    def __new__(cls, redshift):
        # noinspection PyArgumentList
        return float.__new__(cls, redshift)

    def __init__(self, redshift):
        float.__init__(redshift)
