from astropy import cosmology as cosmo

from autogalaxy.cosmology.lensing import LensingCosmology


class LambdaCDM(cosmo.LambdaCDM, LensingCosmology):
    """
    A wrapper for the astropy `LambdaCDM` cosmology class.

    This can be inherited from when creating cosmologies as a `af.Model` object for model-fitting.
    """

    pass


class FlatLambdaCDM(cosmo.FlatLambdaCDM, LensingCosmology):
    """
    A wrapper for the astropy `FlatLambdaCDM` cosmology class.

    This can be inherited from when creating cosmologies as a `af.Model` object for model-fitting.
    """

    pass


class FlatwCDM(cosmo.FlatwCDM, LensingCosmology):
    """
    A wrapper for the astropy `FlatwCDM` cosmology class.

    This can be inherited from when creating cosmologies as a `af.Model` object for model-fitting.
    """

    pass


class Planck15(FlatLambdaCDM, LensingCosmology):
    def __init__(self):
        """
        A wrapper for the astropy `Planck15` cosmology class.

        This can be inherited from when creating cosmologies as a `af.Model` object for model-fitting.
        """
        Planck15 = cosmo.Planck15

        super().__init__(
            H0=Planck15.H0,
            Om0=Planck15.Om0,
            Tcmb0=Planck15.Tcmb0,
            Neff=Planck15.Neff,
            m_nu=Planck15.m_nu,
            Ob0=Planck15.Ob0,
            name=Planck15.name,
        )
