def Planck15():

    """
    A lazy-loading wrapper for the astropy `Planck15` cosmology class.

    The actual class is only created (and astropy imported) when this function is called.
    """

    from astropy import cosmology as cosmo
    from autogalaxy.cosmology.lensing import LensingCosmology

    class _Planck15(cosmo.FlatLambdaCDM, LensingCosmology):
        def __init__(self):
            Planck15_astropy = cosmo.Planck15

            super().__init__(
                H0=Planck15_astropy.H0,
                Om0=Planck15_astropy.Om0,
                Tcmb0=Planck15_astropy.Tcmb0,
                Neff=Planck15_astropy.Neff,
                m_nu=Planck15_astropy.m_nu,
                Ob0=Planck15_astropy.Ob0,
                name=Planck15_astropy.name,
            )

    return _Planck15()