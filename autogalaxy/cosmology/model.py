from astropy import cosmology as cosmo

from autogalaxy.cosmology.lensing import LensingCosmology


class FlatLambdaCDMWrap(cosmo.FlatLambdaCDM, LensingCosmology):
    def __init__(
        self,
        H0: float = 67.66,
        Om0: float = 0.30966,
        Tcmb0: float = 2.7255,
        Neff: float = 3.046,
        m_nu: float = 0.06,
        Ob0: float = 0.04897,
    ):
        super().__init__(
            H0=H0,
            Om0=Om0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu=m_nu,
            Ob0=Ob0,
            name="FlatLambdaCDM",
        )


class FlatwCDMWrap(cosmo.FlatwCDM, LensingCosmology):
    def __init__(
        self,
        H0: float = 67.66,
        Om0: float = 0.30966,
        w0: float = -1.0,
        Tcmb0: float = 2.7255,
        Neff: float = 3.046,
        m_nu: float = 0.06,
        Ob0: float = 0.04897,
    ):
        super().__init__(
            H0=H0,
            Om0=Om0,
            w0=w0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu=m_nu,
            Ob0=Ob0,
            name="FlatwCDM",
        )
