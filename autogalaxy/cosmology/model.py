from astropy import cosmology as cosmo

from autogalaxy.cosmology.lensing import LensingCosmology


class LambdaCDMWrap(cosmo.LambdaCDM, LensingCosmology):
    def __init__(
        self,
        H0: float = 67.66,
        Om0: float = 0.30966,
        Ode0: float = 0.69034,
        Tcmb0: float = 2.7255,
        Neff: float = 3.046,
        m_nu: float = 0.06,
        Ob0: float = 0.04897,
    ):
        """
        A wrapper for the astropy `LambdaCDM` cosmology class, which allows it to be used for modeling such
        that the cosmological parameters are free parameters which can be fitted for.

        The interface of this class is the same as the astropy `LambdaCDM` class, it simply overwrites the
        __init__ method and inherits from it in a way that ensures **PyAutoFit** can compose a model from it
        without issue.

        The class also inherits from `LensingCosmology`, which is a class that provides additional functionality
        for calculating lensing specific quantities in the cosmology.

        Parameters
        ----------
        H0
            The Hubble constant at z=0.
        Om0
            The total matter density at z=0.
        Ode0
            The dark energy density at z=0.
        Tcmb0
            The CMB temperature at z=0.
        Neff
            The effective number of neutrinos.
        m_nu
            The sum of the neutrino masses.
        Ob0
            The baryon density at z=0.
        """
        super().__init__(
            H0=H0,
            Om0=Om0,
            Ode0=Ode0,
            Tcmb0=Tcmb0,
            Neff=Neff,
            m_nu=m_nu,
            Ob0=Ob0,
            name="FlatLambdaCDM",
        )


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
        """
        A wrapper for the astropy `FlatLambdaCDM` cosmology class, which allows it to be used for modeling such
        that the cosmological parameters are free parameters which can be fitted for.

        The interface of this class is the same as the astropy `FlatLambdaCDM` class, it simply overwrites the
        __init__ method and inherits from it in a way that ensures **PyAutoFit** can compose a model from it
        without issue.

        The class also inherits from `LensingCosmology`, which is a class that provides additional functionality
        for calculating lensing specific quantities in the cosmology.

        Parameters
        ----------
        H0
            The Hubble constant at z=0.
        Om0
            The total matter density at z=0.
        Tcmb0
            The CMB temperature at z=0.
        Neff
            The effective number of neutrinos.
        m_nu
            The sum of the neutrino masses.
        Ob0
            The baryon density at z=0.
        """
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
        """
        A wrapper for the astropy `FlatwCDM` cosmology class, which allows it to be used for modeling such
        that the cosmological parameters are free parameters which can be fitted for.

        The interface of this class is the same as the astropy `FlatwCDM` class, it simply overwrites the
        __init__ method and inherits from it in a way that ensures **PyAutoFit** can compose a model from it
        without issue.

        The class also inherits from `LensingCosmology`, which is a class that provides additional functionality
        for calculating lensing specific quantities in the cosmology.

        Parameters
        ----------
        H0
            The Hubble constant at z=0.
        Om0
            The total matter density at z=0.
        w0
            The dark energy equation of state at z=0.
        Tcmb0
            The CMB temperature at z=0.
        Neff
            The effective number of neutrinos.
        m_nu
            The sum of the neutrino masses.
        Ob0
            The baryon density at z=0.
        """
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
