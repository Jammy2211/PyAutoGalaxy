from astropy import cosmology as cosmo

from autogalaxy.cosmology.wrap import FlatLambdaCDM
from autogalaxy.cosmology.wrap import FlatwCDM


class Planck18Om0(FlatLambdaCDM):
    def __init__(self, Om0: float = 0.3075):
        Planck18 = cosmo.Planck18

        super().__init__(
            H0=Planck18.H0,
            Om0=Om0,
            Tcmb0=Planck18.Tcmb0,
            Neff=Planck18.Neff,
            m_nu=Planck18.m_nu,
            Ob0=Planck18.Ob0,
            name=Planck18.name,
        )


class Planck18FlatwCDM(FlatwCDM):
    def __init__(self, Om0: float = 0.3075, w0: float = -1.0):
        Planck18 = cosmo.Planck18

        super().__init__(
            H0=Planck18.H0,
            Om0=Om0,
            w0=w0,
            Tcmb0=Planck18.Tcmb0,
            Neff=Planck18.Neff,
            m_nu=Planck18.m_nu,
            Ob0=Planck18.Ob0,
            name=Planck18.name,
        )
