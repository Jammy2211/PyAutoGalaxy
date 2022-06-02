from astropy import cosmology as cosmo

from autogalaxy.cosmology.lensing import LensingCosmology


class LambdaCDM(cosmo.LambdaCDM, LensingCosmology):

    pass


class FlatLambdaCDM(cosmo.FlatLambdaCDM, LensingCosmology):

    pass


class Planck15(FlatLambdaCDM, LensingCosmology):
    def __init__(self):

        planck15 = cosmo.Planck15

        super().__init__(
            H0=planck15.H0,
            Om0=planck15.Om0,
            Tcmb0=planck15.Tcmb0,
            Neff=planck15.Neff,
            m_nu=planck15.m_nu,
            Ob0=planck15.Ob0,
            name=planck15.name,
        )
