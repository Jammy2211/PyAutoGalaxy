from typing import Tuple

from autogalaxy.profiles.mass.dark.nfw import NFWSph

from astropy import units

import numpy as np
import warnings

from autogalaxy.cosmology.wrap import Planck15


def kappa_s_and_scale_radius(
    virial_mass, concentration, virial_overdens, redshift_object, redshift_source
):
    cosmology = Planck15()

    cosmic_average_density = (
        cosmology.critical_density(redshift_object).to(units.solMass / units.kpc**3)
    ).value

    critical_surface_density = (
        cosmology.critical_surface_density_between_redshifts_solar_mass_per_kpc2_from(
            redshift_0=redshift_object, redshift_1=redshift_source
        )
    )

    kpc_per_arcsec = cosmology.kpc_per_arcsec_from(redshift=redshift_object)

    virial_radius = (
        virial_mass / (virial_overdens * cosmic_average_density * (4.0 * np.pi / 3.0))
    ) ** (
        1.0 / 3.0
    )  # r_vir
    de_c = (
        virial_overdens
        / 3.0
        * (
            concentration**3
            / (np.log(1.0 + concentration) - concentration / (1.0 + concentration))
        )
    )  # rho_c

    scale_radius_kpc = virial_radius / concentration  # scale radius in kpc
    rho_s = cosmic_average_density * de_c  # rho_s
    kappa_s = rho_s * scale_radius_kpc / critical_surface_density  # kappa_s
    scale_radius = scale_radius_kpc / kpc_per_arcsec  # scale radius in arcsec

    return kappa_s, scale_radius, virial_radius


class NFWVirialMassConcSph(NFWSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        virial_mass: float = 1e12,
        concentration: float = 10,
        virial_overdens: float = 200,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        """
        Spherical NFW profile initialized with the mass and concentration of the halo.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        virial_mass
            The virial mass of the dark matter halo.
        concentration
            The concentration of the dark matter halo.
        virial_overdens
            The virial overdensity.
        """
        self.virial_mass = virial_mass
        self.concentration = concentration
        self.virial_overdens = virial_overdens

        (
            kappa_s,
            scale_radius,
            virial_radius,
        ) = kappa_s_and_scale_radius(
            virial_mass=virial_mass,
            concentration=concentration,
            virial_overdens=virial_overdens,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(centre=centre, kappa_s=kappa_s, scale_radius=scale_radius)
