from typing import Tuple

from autogalaxy.profiles.mass.dark.cnfw import cNFWSph

from autogalaxy.profiles.mass.dark import mcr_util

class cNFWMCRLudlowSph(cNFWSph):
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        mass_at_200: float = 1e9,
        f_c = 0.01,
        redshift_object: float = 0.5,
        redshift_source: float = 1.0,
    ):
        self.mass_at_200 = mass_at_200
        self.f_c = f_c
        self.redshift_object = redshift_object
        self.redshift_source = redshift_source

        (
            kappa_s,
            scale_radius,
            core_radius,
            radius_at_200,
        ) = mcr_util.kappa_s_scale_radius_and_core_radius_for_ludlow(
            mass_at_200=mass_at_200,
            f_c=f_c,
            redshift_object=redshift_object,
            redshift_source=redshift_source,
        )

        super().__init__(centre=centre, kappa_s=kappa_s, scale_radius=scale_radius, core_radius=core_radius)